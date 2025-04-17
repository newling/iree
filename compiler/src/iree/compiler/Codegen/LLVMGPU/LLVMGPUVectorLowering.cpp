// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORLOWERINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

//====---------------------------------------------------------------------===//
// Patterns for late vector op lowering.
//====---------------------------------------------------------------------===//

namespace {

// TODO(newling)
//
// Convert as many ops as possible to shape_casts.
//
// broadcast
// extract
// extract_strided_slice
// shape_cast
// transpose
//
// See example in vector_lowering.mlir (transpose -> extract where both are just
// shape_casts).

struct PromoteContractOperands final
    : public vector::MaskableOpRewritePattern<vector::ContractionOp> {
  using MaskableOpRewritePattern::MaskableOpRewritePattern;

  FailureOr<Value>
  matchAndRewriteMaskableOp(vector::ContractionOp contractOp,
                            vector::MaskingOpInterface maskOp,
                            PatternRewriter &rewriter) const override {
    Type operandElType = getElementTypeOrSelf(contractOp.getLhsType());
    Type resultElType = getElementTypeOrSelf(contractOp.getResultType());

    if (operandElType == resultElType) {
      return failure();
    }

    Location loc = contractOp.getLoc();
    Value lhs =
        promoteToElementType(loc, rewriter, contractOp.getLhs(), resultElType);
    Value rhs =
        promoteToElementType(loc, rewriter, contractOp.getRhs(), resultElType);

    auto replacement = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, contractOp.getAcc(), contractOp.getIndexingMaps(),
        contractOp.getIteratorTypes());

    if (!maskOp) {
      return replacement.getResult();
    }
    auto maskedOp = vector::maskOperation(
        rewriter, replacement, maskOp.getMask(), maskOp.getPassthru());
    return maskedOp->getResult(0);
  }

  Value promoteToElementType(Location loc, RewriterBase &rewriter, Value v,
                             Type dstElementType) const {
    Type elementType = getElementTypeOrSelf(v.getType());
    if (elementType == dstElementType)
      return v;

    // vector.contract only allows extension on operands.
    assert(elementType.getIntOrFloatBitWidth() <=
               dstElementType.getIntOrFloatBitWidth() &&
           "vector.contract does not allow truncation of operands");

    Type promotedType = dstElementType;
    if (auto vecType = dyn_cast<VectorType>(v.getType()))
      promotedType = vecType.clone(promotedType);

    if (isa<FloatType>(dstElementType))
      return rewriter.create<arith::ExtFOp>(loc, promotedType, v);
    // For integer types, vector.contract only supports signless integer types
    // and promotion happens via sign extension.
    return rewriter.create<arith::ExtSIOp>(loc, promotedType, v);
  }
};

SmallVector<int64_t> getWithLeadingOnes(VectorType vectorType) {
  SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
  if (vectorType.getRank() > 0) {
    nativeSize.back() = vectorType.getShape().back();
  }
  return nativeSize;
}

SmallVector<int64_t> getWithAllButOneOne(VectorType vectorType) {
  auto rank = vectorType.getRank();
  auto shape = vectorType.getShape();
  auto iter = std::find_if_not(shape.rbegin(), shape.rend(),
                               [](int64_t dim) { return dim == 1; });

  SmallVector<int64_t> nativeSize(rank, 1);
  if (iter != shape.rend()) {
    // Found a non-1 dimension, so we can keep it.
    nativeSize[rank - 1 - std::distance(shape.rbegin(), iter)] = *iter;
  }
  return nativeSize;
}

// A pattern that matches on all elements that have
// `hasElementwiseMappableTraits`, and if there are any size-1 dimensions
// in the result (and inputs), it flattens the inputs with shape_cast
// operations, performs the elementwise operation on the squeezed inputs,
// the uses shape_cast to return to the original shape.

static Operation *cloneOpWithOperandsAndTypes(OpBuilder &builder, Location loc,
                                              Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  return builder.create(loc, op->getName().getIdentifier(), operands,
                        resultTypes, op->getAttrs());
}

struct SqueezeElementwise : public RewritePattern {

  SqueezeElementwise(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op)) {
      return failure();
    }
    if (op->getNumResults() != 1) {
      return failure();
    }
    auto resultType = dyn_cast<VectorType>(op->getResultTypes()[0]);
    if (!resultType) {
      return failure();
    }

    ArrayRef<int64_t> shape = resultType.getShape();
    int64_t nOnes{0};
    for (int64_t dim : shape) {
      if (dim == 1) {
        nOnes++;
      }
    }

    if (nOnes == 0) {
      return failure();
    }

    SmallVector<int64_t> newShape;
    newShape.reserve(shape.size() - nOnes);

    for (int64_t dim : shape) {
      if (dim != 1) {
        newShape.push_back(dim);
      }
    }

    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());

    for (OpOperand &operand : op->getOpOperands()) {
      auto oldVecType = dyn_cast<VectorType>(operand.get().getType());
      if (!oldVecType) {
        newOperands.push_back(operand.get());
      } else {
        assert(oldVecType.getShape() == shape &&
               "operand shape must match result shape");
        VectorType newOperandType =
            VectorType::get(newShape, oldVecType.getElementType());
        auto shapeCast = rewriter.create<vector::ShapeCastOp>(
            op->getLoc(), newOperandType, operand.get());
        newOperands.push_back(shapeCast);
      }
    }

    VectorType newType = VectorType::get(newShape, resultType.getElementType());
    auto newOp = cloneOpWithOperandsAndTypes(rewriter, op->getLoc(), op,
                                             newOperands, newType);

    // cast back up:
    auto castBack = rewriter.create<vector::ShapeCastOp>(
        op->getLoc(), resultType, newOp->getResult(0));

    rewriter.replaceOp(op, castBack.getResult());
    return success();
  }
};

// Similar to the above pattern, but instead of reshaping to have no 1's in the
// shape, this pattern completely flattens all vector operands, so that they are
// all rank-1.
struct ElementwiseToRankOne : public RewritePattern {
  ElementwiseToRankOne(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op)) {
      return failure();
    }
    if (op->getNumResults() != 1) {
      return failure();
    }
    auto resultType = dyn_cast<VectorType>(op->getResultTypes()[0]);
    if (!resultType) {
      return failure();
    }
    if (resultType.getRank() <= 1) {
      return failure();
    }

    auto shape = resultType.getShape();
    SmallVector<int64_t> newShape{resultType.getNumElements()};

    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands());

    for (OpOperand &operand : op->getOpOperands()) {
      auto oldVecType = dyn_cast<VectorType>(operand.get().getType());
      if (!oldVecType) {
        newOperands.push_back(operand.get());
      } else {
        assert(oldVecType.getShape() == shape &&
               "operand shape must match result shape");
        VectorType newOperandType =
            VectorType::get(newShape, oldVecType.getElementType());
        auto shapeCast = rewriter.create<vector::ShapeCastOp>(
            op->getLoc(), newOperandType, operand.get());
        newOperands.push_back(shapeCast);
      }
    }

    VectorType newType = VectorType::get(newShape, resultType.getElementType());
    auto newOp = cloneOpWithOperandsAndTypes(rewriter, op->getLoc(), op,
                                             newOperands, newType);

    // cast back up:
    auto castBack = rewriter.create<vector::ShapeCastOp>(
        op->getLoc(), resultType, newOp->getResult(0));

    rewriter.replaceOp(op, castBack.getResult());
    return success();
  }
};

std::optional<SmallVector<int64_t>> getNativeVectorShape(Operation *op) {

  // Example: elementwise operation `vector<10x4xf32> to vector<10x4xf16>`
  // will be unrolled to 10 elementwise operations on vectors of shape 1x4.
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      return getWithAllButOneOne(vecType);
    }
  }

  // Unroll vector.transpose in all but the inner-most dimension of result.
  // Example: A transpose `vector<2x4xf32> to vector<4x2xf32>` results in 4
  // extract->insert_strided_slice pairs.
  //
  // An alternative is to use `populateVectorTransposeLoweringPatterns`
  // which always creates scalar extract-insert pairs.
  //
  // TODO(newling) reconsider the optimal strategy for this.
  if (auto transposeOp = llvm::dyn_cast<vector::TransposeOp>(op)) {
    return getWithLeadingOnes(transposeOp.getResultVectorType());
  }
  return std::nullopt;
}

struct LLVMGPUVectorLoweringPass final
    : impl::LLVMGPUVectorLoweringPassBase<LLVMGPUVectorLoweringPass> {
  using impl::LLVMGPUVectorLoweringPassBase<
      LLVMGPUVectorLoweringPass>::LLVMGPUVectorLoweringPassBase;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    auto addVectorCanonicalizationPatterns = [](RewritePatternSet &patterns) {
      MLIRContext *ctx = patterns.getContext();
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ExtractStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertOp::getCanonicalizationPatterns(patterns, ctx);
      vector::InsertStridedSliceOp::getCanonicalizationPatterns(patterns, ctx);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, ctx);
      vector::TransposeOp::getCanonicalizationPatterns(patterns, ctx);
    };

    MLIRContext *context = funcOp.getContext();

    // Remove permutation_map, replace with explict broadcast and transpose ops
    // (which we immediately try to canonicalize away).
    {
      RewritePatternSet patterns(context);
      // So far, squeeze is king.
      // patterns.add<SqueezeElementwise>(context);
      // patterns.add<ElementwiseToRankOne>(context);
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    llvm::errs() << "\n\njust after squeezing" << "\n";
    llvm::errs() << funcOp << "\n\n\n";

    // Conversions and unrolls.
    {
      RewritePatternSet patterns(context);
      // Unroll broadcast, leaving rank-1 broadcast.
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      // Unroll gather, leaving rank-1 gathers.
      vector::populateVectorGatherLoweringPatterns(patterns);
      // Unroll create_mask, leaving rank-1 create_masks.
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      // contract to fma.
      vector::populateVectorContractLoweringPatterns(
          patterns, vector::VectorContractLowering::OuterProduct);
      patterns.add<PromoteContractOperands>(context);
      // multi_reduce to arith ops.
      vector::populateVectorMultiReductionLoweringPatterns(
          patterns, vector::VectorMultiReductionLowering::InnerParallel);
      // Remaining vector and other dialect ops have unrolling/lowering handled

      // by 'generic' vector unrolling.
      auto opts = vector::UnrollVectorOptions().setNativeShapeFn(
          [=](auto op) { return getNativeVectorShape(op); });
      vector::populateVectorUnrollPatterns(patterns, opts);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    //   llvm::errs() << "\n\njust before unrolling final" << "\n";
    //   llvm::errs() << funcOp << "\n\n\n";

    //   {

    //     RewritePatternSet patterns(context);
    //     // Remaining vector and other dialect ops have unrolling/lowering
    //     handled
    //     // by 'generic' vector unrolling.
    //     auto opts = vector::UnrollVectorOptions().setNativeShapeFn(
    //         [=](auto op) { return getNativeVectorShape(op); });
    //     vector::populateVectorUnrollPatterns(patterns, opts);
    //     if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
    //       return signalPassFailure();
    //     }
    //   }

    // transfer_read -> load and transfer_write -> store.
    {
      RewritePatternSet patterns(context);
      VectorTransferToSCFOptions vectorToSCFOptions;
      vectorToSCFOptions.enableFullUnroll();
      populateVectorToSCFConversionPatterns(patterns, vectorToSCFOptions);
      vector::populateVectorTransferLoweringPatterns(patterns);
      memref::populateFoldMemRefAliasOpPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Canonicalize.
    {
      RewritePatternSet patterns(context);
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Print the function
    llvm::errs() << "\n\nJust after unroll and canon" << "\n";
    llvm::errs() << funcOp << "\n\n\n";

    //    // Flatten!
    {
      RewritePatternSet patterns(context);
      GreedyRewriteConfig config;
      config.fold = false;

      // TODO(newling) this is very clearly defined set of patterns --

      // energy function that the patterns try to minimize is
      // sum(operations in vector and arith dialects of) energy(op)
      // - energy(shape_cast) = 0
      // - energy(other_op) = sum of ranks of vector operands.

      // IREE uses this as a late stage canonicaliazation before lowering to
      // LLVM, only after unrolling of single-threaded code. Any pattern which
      // decreases this object should be added. Any pattern that increases this
      // objective should definitely not be added to avoid cycles. Any pattern
      // that leaves the energy unchanged -- the energy function can be extended
      // (lexicographically).

      populateFlattenVectorExtractInsertPatterns(patterns);
      patterns.add<ElementwiseToRankOne>(context);
      populateForOpInductionVarShapePatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
        return signalPassFailure();
      }
    }

    // Canonicalize.
    {
      RewritePatternSet patterns(context);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Print the function
    llvm::errs() << "\n\nJust after flatten and canon" << "\n";
    llvm::errs() << funcOp << "\n\n\n";

    if (failed(funcOp->getParentOfType<ModuleOp>().verify())) {
      llvm::errs() << "Module verification failed after flattening\n";
      return signalPassFailure();
    }

    // Less desirable unrolls, delayed till here in case previous
    // canonicalization can eliminate them.
    {
      RewritePatternSet patterns(context);
      // shape_cast to extract-like and insert-like ops.
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    llvm::errs() << "\n\n\njust after shape cast lowering " << "\n";
    llvm::errs() << funcOp << "\n\n\n";

    // Canonicalize.
    {
      RewritePatternSet patterns(context);
      addVectorCanonicalizationPatterns(patterns);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    llvm::errs() << "\n\n\njust after final canon " << "\n";
    llvm::errs() << funcOp << "\n\n\n";
  }
};
} // namespace
} // namespace mlir::iree_compiler
