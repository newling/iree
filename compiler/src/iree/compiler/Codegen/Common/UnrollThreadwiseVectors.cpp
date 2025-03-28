// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===- BubbleUpOrdinalOpPass.cpp -----------------------------------------===//
//
// The workgroup count computation when using slices needs the ordinal
// annotation ops to be bubbled up as much as possible. This pass implements
// patterns to bubble these operations up.
//
//===---------------------------------------------------------------------===//

#include <algorithm>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_UNROLLTHREADWISEVECTORSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct UnrollThreadwiseVectorsPass final
    : impl::UnrollThreadwiseVectorsPassBase<UnrollThreadwiseVectorsPass> {
  using impl::UnrollThreadwiseVectorsPassBase<
      UnrollThreadwiseVectorsPass>::UnrollThreadwiseVectorsPassBase;
  void runOnOperation() override;
};

int getComputeVectorSize(int64_t size) {
  for (int i : {4, 3, 2}) {
    if (size % i == 0)
      return i;
  }
  return 1;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::ReductionOp op) {
  VectorType srcVectorType = op.getSourceVectorType();
  assert(srcVectorType.getRank() == 1); // Guaranteed by semantics
  int64_t vectorSize = getComputeVectorSize(srcVectorType.getDimSize(0));
  return {vectorSize};
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::TransposeOp op) {
  VectorType vectorType = op.getResultVectorType();
  SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
  nativeSize.back() = getComputeVectorSize(vectorType.getShape().back());
  return nativeSize;
}

std::optional<SmallVector<int64_t>> getNativeVectorShape(Operation *op) {
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = dyn_cast<VectorType>(op->getResultTypes()[0])) {
      SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = getComputeVectorSize(vecType.getShape().back());
      return nativeSize;
    }
  }

  return TypeSwitch<Operation *, std::optional<SmallVector<int64_t>>>(op)
      .Case<vector::ReductionOp, vector::TransposeOp>(
          // vector::MultiDimReductionOp>(
          [](auto typedOp) { return getNativeVectorShapeImpl(typedOp); })
      .Default([](Operation *) { return std::nullopt; });
}

// For now, we detect vector.transpose ops that have at most 1 element in the
// shape that is not size 1.
class TransposeToShapeCast final
    : public OpRewritePattern<vector::TransposeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto vectorType = dyn_cast<VectorType>(transposeOp.getType());
    if (!vectorType) {
      return failure();
    }
    auto shape = vectorType.getShape();
    auto nNonOne = std::count_if(shape.begin(), shape.end(),
                                 [](int64_t i) { return i != 1; });

    if (nNonOne > 1) {
      return failure();
    }

    // Replace with a shape_cast:
    auto newTransposeOp = rewriter.createOrFold<vector::ShapeCastOp>(
        transposeOp.getLoc(), vectorType, transposeOp.getVector());
    rewriter.replaceOp(transposeOp, newTransposeOp);
    return success();
  }
};

/////////////////////////////////
/// Patterns on vector.extract //
/////////////////////////////////

// Example 1:
// %dst = vector.extract %src[0, 0, 0, 0, 0] : vector<8xf16> from
//                                             vector<4x1x1x1x1x8xf16>
// to
//
// %src_flat = vector.shape_cast %src : vector<4x1x1x1x1x8xf16> to
//                                      vector<4x8xf16>
// %dst = vector.extract %src_flat[0, 0, 0, 0] : vector<8xf16> from
//                                               vector<4x8xf16>
//
//
// Example 2:
// %dst = vector.extract %arg1[1, 0, 0] : f32 from
//                                        vector<2x1x1xf32>
// to
//
// %src_flat = vector.shape_cast %arg1 : vector<2x1x1xf32> to
//                                       vector<2xf32>
// %dst = vector.extract %src_flat[1] : f32 from
//                                      vector<2xf32>
//
class RankReduceExtractSrc final : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {

    auto positions = extractOp.getMixedPosition();
    VectorType vectorType = extractOp.getSourceVectorType();
    auto shape = vectorType.getShape();

    int dstRank{0};
    if (extractOp.getType().isIntOrIndexOrFloat()) {
      dstRank = 0;
    } else if (auto extractedType = dyn_cast<ShapedType>(extractOp.getType())) {
      dstRank = extractedType.getRank();
    } else {
      llvm::errs() << "extract op is " << extractOp << "\n";
      llvm::errs() << "extract op type is " << extractOp.getType() << "\n";
      llvm::errs() << "is int or index or float ? "
                   << extractOp.getType().isIntOrIndexOrFloat() << "\n";
      assert(false && "haha");
      return failure();
    }

    // Pattern only applicable of the destination rank is 2+ less than than
    // the source rank.
    int64_t rankDifference = vectorType.getRank() - dstRank;
    // TODO(newling) oh, not strictly greater than 
    assert(rankDifference >= 0);
    if (rankDifference == 1) {
      return failure();
    }

    int64_t nbDimsToCollapse = rankDifference;
    int64_t newPosition{0};
    int64_t outerDim{1};
    for (int i = nbDimsToCollapse; i > 0; --i) {
      OpFoldResult positionFoldRes = positions[i - 1];
      auto positionAttr = dyn_cast<Attribute>(positionFoldRes);
      if (!positionAttr) {
        return failure();
      }
      int64_t position = cast<IntegerAttr>(positionAttr).getInt();
      newPosition += position * outerDim;
      outerDim *= vectorType.getDimSize(i - 1);
    }
    SmallVector<OpFoldResult> newPositions;
    newPositions.reserve(shape.size() + 1 - nbDimsToCollapse);
    newPositions.push_back(rewriter.getI64IntegerAttr(newPosition));
    newPositions.insert(newPositions.end(), positions.begin() + nbDimsToCollapse,
                      positions.end());

    SmallVector<int64_t> newShape;
    newShape.reserve(shape.size() + 1 - nbDimsToCollapse);
    newShape.push_back(outerDim);
    newShape.insert(newShape.end(), shape.begin() + nbDimsToCollapse,
                    shape.end());

    VectorType newType = VectorType::get(newShape, vectorType.getElementType());

    // Cast the source vector to the new shape:
    auto downRank = rewriter.createOrFold<vector::ShapeCastOp>(
        extractOp.getLoc(), newType, extractOp.getVector());

    // Perform the extract on the new vector:
    auto replacement = rewriter.create<vector::ExtractOp>(extractOp.getLoc(),
                                                          downRank, newPositions);

    rewriter.replaceOp(extractOp, replacement);
    return success();
  }
};

// Example 1:
// %dst = vector.extract %src[0] : vector<1x8xf16> from
//                                 vector<2x1x8xf16>
// to
//
// %dst_flat = vector.extract %src[0, 0] : vector<8xf16> from
//                                         vector<2x1x8xf16>
// %dst = vector.shape_cast %dst_flat : vector<8xf16> to
//                                      vector<1x8xf16>
//
// Example 2:
//
// %dst = vector.extract %src[0] : vector<1x1xf16> from
//                                 vector<2x1x1xf16>
// to
// 
// %dst_flat = vector.extract %src[0, 0, 0] : f16 from
//                                            vector<2x1x1xf16>
// %dst = vector.broadcast %dst_flat : f16 to
//                                     vector<1x1xf16>
//
// The above 'broadcast' is because of issues around scalar/rank-0. 
class RankReduceExtractDst final : public OpRewritePattern<vector::ExtractOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {

    auto type = dyn_cast<VectorType>(extractOp.getType());
    if (!type) {
      return failure();
    }
    auto shape = type.getShape();
    if (shape.empty()) {
      return failure();
    }

    // Sanity check that we're going the right way here. 
    // TODO(newling) oh, extract can have same rank for input and output. 
    assert(shape.size() <= extractOp.getSourceVectorType().getRank());
    uint64_t nbOnes{0};
    for (auto [i, dim] : llvm::enumerate(shape)) {
      if (dim > 1) {
        break;
      }
      ++nbOnes;
    }
    if (nbOnes == 0) {
      return failure();
    }
    SmallVector<OpFoldResult> position = extractOp.getMixedPosition();
    auto zeroFoldResult = rewriter.getIndexAttr(0);
    for (uint64_t i = 0; i < nbOnes; i++) {
      position.push_back(zeroFoldResult);
    }

    // Create a new extractOp:
    auto newExtractOp = rewriter.create<vector::ExtractOp>(
        extractOp.getLoc(), extractOp.getVector(), position);

    if (nbOnes == shape.size()) {
      // vector.broadcast to rank-1:
      auto repl = rewriter.createOrFold<vector::BroadcastOp>(
          extractOp.getLoc(), extractOp.getType(), newExtractOp.getResult());
      rewriter.replaceOp(extractOp, repl);
    } else {
      auto repl = rewriter.createOrFold<vector::ShapeCastOp>(
          extractOp.getLoc(), type, newExtractOp.getResult());
      rewriter.replaceOp(extractOp, repl);
    }
    return success();
  }
};

// A Pattern to replace vector.extract op where the number of elements of the
// source and destination are the same, with a vector.shape_cast.
class ExtractWithSameSize final : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType =
        dyn_cast<VectorType>(extractOp.getVector().getType());
    auto destVectorType = dyn_cast<VectorType>(extractOp.getType());
    if (!sourceVectorType || !destVectorType) {
      return failure();
    }
    if (!sourceVectorType.hasStaticShape() ||
        !destVectorType.hasStaticShape()) {
      return failure();
    }
    if (sourceVectorType.getNumElements() != destVectorType.getNumElements()) {
      return failure();
    }

    auto shapeCastRepl = rewriter.createOrFold<vector::ShapeCastOp>(
        extractOp.getLoc(), destVectorType, extractOp.getVector());

    rewriter.replaceOp(extractOp, shapeCastRepl);

    return success();
  }
};

// Our overall goal is to make all vector operands as low-rank as possible,
// by using shape_cast, and hoping that the shape_casts generated across
// patterns cancel.
//
// Below we have 2 simple patterns for reducing the rank of the src and dst of
// insert_strided_slice.
//
// Example of an insert_strided_slice op:
// %out = vector.insert_strided_slice %src, %dst
//       {offsets = [2, 0, 0, 0, 0, 0],
//        strides = [1, 1]} :
//          vector<1x8xf16> into
//          vector<4x1x1x1x1x8xf16>
//
// This will get converted by the 2 patterns to:
//
// %src_flat = vector.shape_cast %src : vector<1x8xf16> to vector<8xf16>
// %dst_flat = vector.shape_cast %dst : vector<4x1x1x1x1x8xf16> to
//                                                       vector<32xf16> 
// %out_flat = vector.insert_strided_slice %src_flat, %dst_flat
//                   {offsets = [16], strides = [1]} : vector<8xf16> into
//                   vector<32xf16>
// %out = vector.shape_cast %out_flat : vector<32xf16> to
// vector<4x1x1x1x1x8xf16>
//
//
// Another example of an insert_strided_slice op:
//
// %out = vector.insert_strided_slice %src, %dst
//       {offsets = [1, 0, 2, 1, 0, 0],
//        strides = [1]} :
//          vector<1xf32> into
//          vector<4x1x4x2x1x1xf32>
//
// This will get converted to:
//
// %dst_flat = vector.shape_cast %dst : vector<4x1x4x2x1x1xf32> to
// vector<32xf32> %out_flat = vector.insert_strided_slice %src, %dst_flat
//                   {offsets = [13], strides = [1]} : vector<1xf32> into
//                   vector<32xf32>
// %out = vector.shape_cast %out_flat : vector<32xf32> to
// vector<4x1x4x2x1x1xf32>
//
// Note that we cannot flatten %src down to rank-0 in the above case, because
// insert_strided_slice cannot take a rank-0 vector, or a raw scalar (f32).
//
// The first pattern below handles the flattening of src (just removes leading
// ones).

// NOTE: this pattern is probably the same as
// CastAwayInsertStridedSliceLeadingOneDim
class RankReduceInsertStridedSliceSrc final
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp insertOp,
                                PatternRewriter &rewriter) const override {

    VectorType sourceVectorType = insertOp.getSourceVectorType();
    int64_t sourceVectorRank = sourceVectorType.getRank();
    if (sourceVectorRank <= 1) {
      return failure();
    }

    if (!sourceVectorType.hasStaticShape()) {
      return failure();
    }
    ArrayRef<int64_t> sourceShape = sourceVectorType.getShape();

    // Compute the number of leading dimensions to drop from the source vector.
    // At most `rank - 1` leading dimensions will be dropped, in other words the
    // rank of the source vector will be at least 1. This is because the
    // insert_strided_slice op is not implemented to take a rank-0 vector, or a
    // scalar, as the source.
    int64_t nbDimsToDrop = 0;
    while (nbDimsToDrop + 1 < sourceVectorRank &&
           (sourceVectorType.getDimSize(nbDimsToDrop) == 1)) {
      nbDimsToDrop++;
    }
    if (nbDimsToDrop == 0) {
      return failure();
    }

    // Drop the leading dimensions from the source vector.
    ArrayRef<int64_t> newShape = sourceShape.drop_front(nbDimsToDrop);
    VectorType newType =
        VectorType::get(newShape, sourceVectorType.getElementType());
    Value shapeCast = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), newType, insertOp.getSource());

    // Create a new insert_strided_slice op with the new source vector.
    ArrayRef<Attribute> newStrides =
        insertOp.getStrides().getValue().drop_front(nbDimsToDrop);
    auto replacement = rewriter.create<vector::InsertStridedSliceOp>(
        insertOp.getLoc(), shapeCast, insertOp.getDest(), insertOp.getOffsets(),
        rewriter.getArrayAttr(newStrides));
    rewriter.replaceOp(insertOp, replacement);
    return success();
  }
};

class RankReduceInsertStridedSliceDst final
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp insertOp,
                                PatternRewriter &rewriter) const override {

    VectorType vectorType = insertOp.getDestVectorType();
    auto shape = vectorType.getShape();
    auto offsets = insertOp.getOffsets().getValue();
    int64_t rankDifference =
        vectorType.getRank() - insertOp.getSourceVectorType().getRank();
    if (rankDifference <= 0) {
      return failure();
    }
    int64_t nbDimsToCollapse = 1 + rankDifference;
    int64_t newOffset{0};
    int64_t outerDim{1};
    for (int i = nbDimsToCollapse; i > 0; --i) {
      Attribute offsetAttr = offsets[i - 1];
      auto offsetIntAttr = dyn_cast<IntegerAttr>(offsetAttr);
      if (!offsetIntAttr) {
        return failure();
      }
      int offset = offsetIntAttr.getInt();
      newOffset += offset * outerDim;
      outerDim *= vectorType.getDimSize(i - 1);
    }
    SmallVector<Attribute> newOffsets;
    newOffsets.reserve(shape.size() + 1 - nbDimsToCollapse);
    newOffsets.push_back(rewriter.getI64IntegerAttr(newOffset));
    newOffsets.insert(newOffsets.end(), offsets.begin() + nbDimsToCollapse,
                      offsets.end());

    SmallVector<int64_t> newShape;
    newShape.reserve(shape.size() + 1 - nbDimsToCollapse);
    newShape.push_back(outerDim);
    newShape.insert(newShape.end(), shape.begin() + nbDimsToCollapse,
                    shape.end());

    VectorType newType = VectorType::get(newShape, vectorType.getElementType());

    // Use vector.shape_cast on largeination vector to get a vector of type
    // 'newType'.
    auto downRank = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), newType, insertOp.getDest());

    // Now, create a new insert_strided_slice op with the new largeination
    // vector.
    auto replacement = rewriter.create<vector::InsertStridedSliceOp>(
        insertOp.getLoc(), insertOp.getSource(), downRank,
        rewriter.getArrayAttr(newOffsets), insertOp.getStrides());

    // Shape cast back to the original shape:
    auto upRank = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), vectorType, replacement.getResult());

    rewriter.replaceOp(insertOp, upRank);
    return success();
  }
};


// Example:
//
// %dst = vector.extract_strided_slice %src
//     {offsets = [1, 0, 0, 0, 0, 4], 
//      sizes = [1, 1, 1, 1, 1, 4], 
//      strides = [1, 1, 1, 1, 1, 1]} : 
//            vector<4x1x1x1x1x8xf16> to vector<1x1x1x1x1x4xf16>
//
// becomes, first with the pattern RankReduceExtractStridedSliceDst:
//
// Dude this destination is contiguous, just ram it our for now. 
//
//
class RankReduceContiguousExtractStridedSlice final
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult
  matchAndRewrite(vector::ExtractStridedSliceOp extractStridedSliceOp,
                  PatternRewriter &rewriter) const override {

    
    VectorType dstType = extractStridedSliceOp.getType();
    auto dstShape = dstType.getShape();
    if (dstShape.size() <= 1) {
      return failure();
    }
    // For now, only consider case where only final dim is not 1.
    for (int i = 0; i < dstShape.size() - 1; i++) {
      if (dstShape[i] != 1) {
        return failure();
      }
    }

    // TODO(newling) assert that all strides are 1. 
    // (void)strides;

    SmallVector<int64_t> offsets;
    offsets.reserve(dstShape.size());
    auto offsetsAttr = extractStridedSliceOp.getOffsets().getValue();
    for (auto offsetAttr : offsetsAttr) {
      auto offsetIntAttr = dyn_cast<IntegerAttr>(offsetAttr);
      if (!offsetIntAttr) {
        return failure();
      }
      offsets.push_back(offsetIntAttr.getInt());
    }
    while (offsets.size() < dstShape.size()) {
      offsets.push_back(0);
    }

    int64_t globalOffset{0};
    int64_t stride{1};
    for (int64_t i = dstShape.size() - 1; i >= 0; i--) {
      globalOffset += offsets[i] * stride;
      stride *= dstShape[i];
    }

    // Step 1: collapse the src vector down to rank-1:
    auto srcVectorType = extractStridedSliceOp.getSourceVectorType();
    auto flatSrcVectorType = VectorType::get({srcVectorType.getNumElements()},
                                             srcVectorType.getElementType());
    auto flatSrc = rewriter.createOrFold<vector::ShapeCastOp>(
        extractStridedSliceOp.getLoc(), flatSrcVectorType,
        extractStridedSliceOp.getVector());

    int64_t dstNbElms = dstType.getNumElements();

    // Step 2: extract a flat vector from the flat src, using the new offset.
    auto newExtractStridedSliceOp =
        rewriter.create<vector::ExtractStridedSliceOp>(
            extractStridedSliceOp.getLoc(), flatSrc,
            SmallVector<int64_t>{globalOffset}, SmallVector<int64_t>{dstNbElms},
            SmallVector<int64_t>{1});

    // Step 3: shape cast back to the original shape:
    auto newDst = rewriter.createOrFold<vector::ShapeCastOp>(
        extractStridedSliceOp.getLoc(), dstType,
        newExtractStridedSliceOp.getResult());

    rewriter.replaceOp(extractStridedSliceOp, newDst);

    return success();
  }
};

} // namespace

// Patterns, where the fixed point is defined by energy function E (to be
// minimized):
//
// vector.shape_cast is free (can have as many as you like, don't effect E)
// low-rank operands and results are better (they reduce E).
//
void UnrollThreadwiseVectorsPass::runOnOperation() {

  Operation *op = getOperation();
  MLIRContext *context = &getContext();

  // Make the operations act on vectors with at most 1 dimension that is not 1.
  // Follow-up passes will try to manipulate the dimensions of size 1 away.
  {
    RewritePatternSet patterns(context);
    auto opts =
        vector::UnrollVectorOptions().setNativeShapeFn(getNativeVectorShape);
    vector::populateVectorUnrollPatterns(patterns, opts);
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vector::VectorMultiReductionLowering::InnerParallel);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // General canonicalization:
  {
    RewritePatternSet patterns(context);
    vector::InsertOp::getCanonicalizationPatterns(patterns, context);
    vector::ExtractOp::getCanonicalizationPatterns(patterns, context);
    vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // Reduce rank of ops with vector.shape_cast, and hope shape_casts cancel.
  {
    RewritePatternSet patterns(context);
    // Must disable folding, to avoid for example destruction of
    // foldExtractFromShapeCast
    GreedyRewriteConfig config;
    config.fold = false;
    vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    patterns.insert<
        // vector.extract rank reducers:
        RankReduceExtractDst, RankReduceExtractSrc,
        // vector.insert_strided_slice rank reducers:
        /* RankReduceInsertStridedSliceSrc <-- upstream */
        RankReduceInsertStridedSliceDst,
        // vector.extract_strided_slice rank reducers:
        RankReduceContiguousExtractStridedSlice,

        // Others
        ExtractWithSameSize, TransposeToShapeCast>(context);
    vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);
    vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(op, std::move(patterns), config))) {
      return signalPassFailure();
    }
  }
}

} // namespace mlir::iree_compiler
