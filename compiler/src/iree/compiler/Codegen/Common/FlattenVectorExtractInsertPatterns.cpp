// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===-- FlattenVectorExtractInsertPatterns.cpp --------------------===//
//
//===--------------------------------------------------------------===//

#include <algorithm>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTFLATTENVECTOREXTRACTINSERTPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Get the global offset/position of the extract op in a flat vector.
FailureOr<int64_t> getGlobalOffset(PatternRewriter &rewriter,
                                   SmallVector<OpFoldResult> positions,
                                   VectorType largeType) {

  // Ensure offsets/positions is of the same rank as the large vector
  // by prepending zeros.
  assert(positions.size() <= largeType.getRank() &&
         "positions size should be less than or equal to large vector rank");
  IntegerAttr zeroFoldResult = rewriter.getIndexAttr(0);
  while (positions.size() < largeType.getRank()) {
    positions.push_back(zeroFoldResult);
  }

  // Compute the global offset. This is essentially the offset of the element
  // at index `positions` in the vector with row-major striding.
  int64_t stride{1};
  int64_t globalOffset{0};
  for (int64_t i = largeType.getRank() - 1; i >= 0; i--) {
    OpFoldResult positionFoldRes = positions[i];
    Attribute positionAttr = dyn_cast<Attribute>(positionFoldRes);
    if (!positionAttr) {
      return failure();
    }
    int64_t position = cast<IntegerAttr>(positionAttr).getInt();
    globalOffset += position * stride;
    stride *= largeType.getDimSize(i);
  }
  return globalOffset;
}

FailureOr<int64_t> getNumberOfElements(Type type) {
  if (type.isIntOrIndexOrFloat()) {
    return 1;
  } else if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.getNumElements();
  }
  return failure();
}

// Convert vector.extract ops with inputs of rank > 1:
//
// If the result has a single element, convert to an extract op on
// a vector of rank-1 with a scalar result.
//
// Otherwise, convert to an extract_strided_slice op on a vector of rank-1.
//
// In other words: create an extract or extract_strided_slice with the lowest
// possible value of (rank input + rank output).
class ConvertExtractOpToRankOneOp final
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {

    TypedValue<VectorType> large = extractOp.getVector();
    VectorType largeType = large.getType();

    SmallVector<OpFoldResult> positions = extractOp.getMixedPosition();
    Location loc = extractOp.getLoc();

    if (largeType.getRank() == 1) {
      return failure();
    }
    VectorType flatLargeType = VectorType::get({largeType.getNumElements()},
                                               largeType.getElementType());

    Type smallType = extractOp.getType();
    FailureOr<int64_t> maybeNumberElementsSmall =
        getNumberOfElements(smallType);
    if (failed(maybeNumberElementsSmall)) {
      return failure();
    }
    int64_t numberElementsSmall = maybeNumberElementsSmall.value();

    FailureOr<int64_t> maybeGlobalOffset =
        getGlobalOffset(rewriter, positions, largeType);
    if (failed(maybeGlobalOffset)) {
      return failure();
    }
    int64_t globalOffset = maybeGlobalOffset.value();

    auto replacement = [&]() -> Value {
      Value flatLarge =
          rewriter.createOrFold<vector::ShapeCastOp>(loc, flatLargeType, large);

      // If the extract op's output is a scalar, create another extract to
      // scalar.
      if (smallType.isIntOrIndexOrFloat()) {
        return rewriter.create<vector::ExtractOp>(
            extractOp.getLoc(), flatLarge, SmallVector<int64_t>{globalOffset});
      }

      auto strided = rewriter.create<vector::ExtractStridedSliceOp>(
          extractOp.getLoc(), flatLarge, SmallVector<int64_t>{globalOffset},
          SmallVector<int64_t>{numberElementsSmall}, SmallVector<int64_t>{1});

      return rewriter.createOrFold<vector::ShapeCastOp>(extractOp.getLoc(),
                                                        smallType, strided);
    }();

    rewriter.replaceOp(extractOp, replacement);
    return success();
  }
};

// Convert vector.insert where the destination is rank > 1.
//
// If the source is a scalar, convert to a vector.insert op into a vector of
// rank-1.
//
// Otherwise, convert to a vector.insert_strided_slice op into a vector of
// rank-1.
//
// In other words: create an insert or insert_strided_slice with the lowest
// possible value of (rank source + rank destination).
class ConvertInsertOpToRankOneOp final
    : public OpRewritePattern<vector::InsertOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::InsertOp insertOp,
                                PatternRewriter &rewriter) const override {

    TypedValue<VectorType> large = insertOp.getDest();
    VectorType largeType = large.getType();
    if (largeType.getRank() == 1) {
      return failure();
    }

    SmallVector<OpFoldResult> positions = insertOp.getMixedPosition();
    Location loc = insertOp.getLoc();

    Value small = insertOp.getValueToStore();
    Type smallType = insertOp.getValueToStoreType();
    auto maybeNumberElementsSmall = getNumberOfElements(smallType);
    if (failed(maybeNumberElementsSmall)) {
      return failure();
    }
    int64_t numberElementsSmall = maybeNumberElementsSmall.value();

    auto maybeGlobalOffset = getGlobalOffset(rewriter, positions, largeType);
    if (failed(maybeGlobalOffset)) {
      return failure();
    }
    int64_t globalOffset = maybeGlobalOffset.value();

    // Rank-1 source.
    VectorType flatLargeType = VectorType::get({largeType.getNumElements()},
                                               largeType.getElementType());
    auto flatLarge =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, flatLargeType, large);

    Value updated = [&]() -> Value {
      if (smallType.isSignlessIntOrFloat()) {
        return rewriter.create<vector::InsertOp>(
            loc, small, flatLarge, SmallVector<int64_t>{globalOffset});
      }
      VectorType flatSmallType =
          VectorType::get({numberElementsSmall}, largeType.getElementType());
      auto flatSmall =
          rewriter.createOrFold<vector::ShapeCastOp>(loc, flatSmallType, small);
      return rewriter.create<vector::InsertStridedSliceOp>(
          insertOp.getLoc(), flatSmall, flatLarge,
          SmallVector<int64_t>{globalOffset}, SmallVector<int64_t>{1});
    }();

    Value replacement = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), largeType, updated);

    rewriter.replaceOp(insertOp, replacement);
    return success();
  }
};

bool isStrideless(VectorType inputType, VectorType resultType) {
  bool sliceSeen = false;
  int x0 = 0;
  int x1 = inputType.getRank() - 1;
  int64_t delta = resultType.getRank() - inputType.getRank();
  // Increase x0 until we find a dimension that is not 1.
  while (x0 < x1 && inputType.getDimSize(x0) == 1)
    ++x0;
  // Decrease x1 until we assert this is actually strided.
  while (x1 >= x0) {
    int64_t outSize = resultType.getDimSize(x1 + delta);
    if (outSize > 1) {
      int64_t inSize = inputType.getDimSize(x1);
      bool isSlice = outSize != inSize;
      if (sliceSeen && isSlice) {
        return false;
      }
      if (isSlice) {
        sliceSeen = true;
      }
    }
    --x1;
  }

  return true;
}

int64_t getGlobalOffset(ArrayAttr offsets, VectorType type) {

  assert(offsets && "Expected offsets to be non-null");

  int64_t globalOffset = 0;
  int64_t stride = 1;
  for (int i = type.getRank() - 1; i >= 0; --i) {
    auto dimSize = type.getDimSize(i);
    if (dimSize > 1) {
      Attribute offset = offsets[i];
      assert(isa<IntegerAttr>(offset) && "Expected IntegerAttr");
      int64_t offsetValue = cast<IntegerAttr>(offset).getInt();
      globalOffset += offsetValue * stride;
      stride *= dimSize;
    }
  }
  return globalOffset;
}

/// If the operation is not already rank-1 to rank-1, and it can be written as
/// rank-1 to rank-1, do so.
class FlattenInsertStridedSliceOp final
    : public OpRewritePattern<vector::InsertStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::InsertStridedSliceOp insertOp,
                                PatternRewriter &rewriter) const override {

    VectorType resultType = insertOp.getType();
    TypedValue<VectorType> input = insertOp.getValueToStore();
    VectorType inputType = input.getType();

    ArrayAttr strides = insertOp.getStrides();
    (void)strides;
    // TODO(newling) Confirm all strides are 1.

    if (inputType.getRank() <= 1 && resultType.getRank() <= 1) {
      return failure();
    }

    if (!isStrideless(inputType, resultType)) {
      return failure();
    }

    int64_t globalOffset = getGlobalOffset(insertOp.getOffsets(), resultType);

    VectorType flatInputType = VectorType::get({inputType.getNumElements()},
                                               inputType.getElementType());
    VectorType flatOutputType = VectorType::get({resultType.getNumElements()},
                                                resultType.getElementType());

    auto flatValueToInsert = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), flatInputType, input);

    auto flatValueToInsertInto = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), flatOutputType, insertOp.getDest());

    auto replacement = rewriter.create<vector::InsertStridedSliceOp>(
        insertOp.getLoc(), flatValueToInsert, flatValueToInsertInto,
        SmallVector<int64_t>{globalOffset}, SmallVector<int64_t>{1});

    auto expanded = rewriter.createOrFold<vector::ShapeCastOp>(
        insertOp.getLoc(), resultType, replacement);

    rewriter.replaceOp(insertOp, expanded);

    return success();
  }
};

class FlattenExtractStridedSliceOp final
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp extractOp,
                                PatternRewriter &rewriter) const override {

    VectorType resultType = extractOp.getType();
    VectorType inputType = extractOp.getSourceVectorType();

    if (inputType.getRank() <= 1 && resultType.getRank() <= 1) {
      return failure();
    }

    if (!isStrideless(inputType, resultType)) {
      return failure();
    }

    int64_t globalOffset = getGlobalOffset(extractOp.getOffsets(), inputType);

    VectorType flatInputType = VectorType::get({inputType.getNumElements()},
                                               inputType.getElementType());

    auto flatInput = rewriter.createOrFold<vector::ShapeCastOp>(
        extractOp.getLoc(), flatInputType, extractOp.getOperand());

    auto replacement = rewriter.create<vector::ExtractStridedSliceOp>(
        extractOp.getLoc(), flatInput, SmallVector<int64_t>{globalOffset},
        SmallVector<int64_t>{resultType.getNumElements()},
        SmallVector<int64_t>{1});

    auto flatOut = rewriter.createOrFold<vector::ShapeCastOp>(
        extractOp.getLoc(), resultType, replacement);

    rewriter.replaceOp(extractOp, flatOut);

    return success();
  }
};

struct TestFlattenVectorExtractInsertPass final
    : public impl::TestFlattenVectorExtractInsertPassBase<
          TestFlattenVectorExtractInsertPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // Folding must be disable when applying these patterns, because the
    // vector.extract op 'folds' extract(shape_cast(x)) -> extract(x), but
    // the pattern ConvertExtractOpToRankOneOp converts
    // extract(x) to extract(shape_cast(x)) in some cases.
    GreedyRewriteConfig config;
    config.fold = false;
    populateFlattenVectorExtractInsertPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};

struct TransposeToShapeCast final
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    // TODO.
    return success();
  }

} // namespace


void populateConvertToShapeCastPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit = 1) {
  patterns.insert<TransposeToShapeCast>(patterns.getContext(), benefit);
}

// These patterns bla
void populateFlattenVectorExtractInsertPatterns(RewritePatternSet &patterns,
                                                PatternBenefit benefit) {

  populateConvertToShapeCastPatterns(patterns, benefit);
  patterns.insert<ConvertExtractOpToRankOneOp, ConvertInsertOpToRankOneOp,
                  FlattenExtractStridedSliceOp, FlattenInsertStridedSliceOp>(
      patterns.getContext(), benefit);
}

} // namespace mlir::iree_compiler
