// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUVECTORLOWERINGPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

//====---------------------------------------------------------------------===//
// Patterns for late vector op lowering.
//====---------------------------------------------------------------------===//

namespace {

int getComputeVectorSize(int64_t size) {
  for (int i : {4, 3, 2}) {
    if (size % i == 0)
      return i;
  }
  return 1;
}

int getMemoryVectorSize(Value source, Type scalarType, int64_t size) {
  int bitwidth = scalarType.getIntOrFloatBitWidth();
  while (auto sliceOp = source.getDefiningOp<tensor::ExtractSliceOp>())
    source = sliceOp.getSource();
  if (!matchPattern(source, m_Constant())) {
    // If we are not reading from a constant array that is embedded in the
    // kernel, try to use a large vector size matching the bitwidth to read in
    // 128-bit chunks. This helps with memory access performance. Such vector
    // sizes are not native in SPIR-V though; this relies on following passes to
    // bitcast them to 32-bit 4-element vectors to be valid.
    if (bitwidth <= 8 && size % 16 == 0)
      return 16;
    if (bitwidth <= 16 && size % 8 == 0)
      return 8;
  }
  if (bitwidth <= 32 && size % 4 == 0)
    return 4;
  return size % 2 == 0 ? 2 : 1;
}

SmallVector<int64_t> getNativeVectorShapeImpl(VectorTransferOpInterface op) {
  auto vecType = op.getVectorType();
  SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
  for (const auto &[index, dim] :
       llvm::enumerate(op.getPermutationMap().getResults())) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(dim)) {
      if (dimExpr.getPosition() == op.getPermutationMap().getNumDims() - 1) {
        nativeSize[index] =
            getMemoryVectorSize(op.getSource(), vecType.getElementType(),
                                vecType.getShape()[index]);
      }
    }
  }
  return nativeSize;
}

Operation *stripElementBitPatternPreservingParents(Value op) {
  while (Operation *parentOp = op.getDefiningOp()) {
    Value source =
        TypeSwitch<Operation *, Value>(parentOp)
            .Case<vector::BroadcastOp>([](vector::BroadcastOp broadcast) {
              return broadcast.getVector();
            })
            .Case<vector::ExtractOp, vector::ExtractElementOp,
                  vector::ExtractStridedSliceOp>(
                [](auto extract) { return extract.getVector(); })
            .Case<vector::InsertOp, vector::InsertElementOp,
                  vector::InsertStridedSliceOp>(
                [](auto insert) { return insert.getSource(); })
            .Case<vector::TransposeOp>([](vector::TransposeOp transpose) {
              return transpose.getVector();
            })
            .Default([](Operation *) { return nullptr; });

    if (!source)
      break;
    op = source;
  }

  return op.getDefiningOp();
}

/// Returns true when |op| has the i32 element type that is likely to be result
/// of a zero/sign extension from i8.
bool mayExtI8ToI32(Value op) {
  if (!getElementTypeOrSelf(op.getType()).isInteger(32))
    return false;

  // Look through vector operations created by vector unrolling patterns,
  // hoping to find a zero/sign extension op. Note that we do not need to find
  // the exact definition for |op| as the final extension will be matched by
  // other patterns -- we only need a good enough proxy to know that one is
  // likely to be found after canonicalization.
  // TODO(#12543): Implement integer narrowing patterns to be able to tell for
  // sure.
  Operation *def = stripElementBitPatternPreservingParents(op);
  Type inTy;

  if (auto ext = dyn_cast_or_null<arith::ExtSIOp>(def)) {
    inTy = getElementTypeOrSelf(ext.getIn().getType());
  } else if (auto ext = dyn_cast_or_null<arith::ExtUIOp>(def)) {
    inTy = getElementTypeOrSelf(ext.getIn().getType());
  } else {
    return false;
  }

  return inTy.isInteger(8);
}

/// Succeeds when |contract| is a i32 matmul whose LHS and RHS operands may be
/// result of zero/sign extension of i8 inputs.
LogicalResult detectI8ToI32Matmul(vector::ContractionOp contract) {
  if (contract.getKind() != vector::CombiningKind::ADD)
    return failure();

  if (!mayExtI8ToI32(contract.getLhs()) || !mayExtI8ToI32(contract.getRhs()))
    return failure();

  ArrayRef<Attribute> iteratorTypes = contract.getIteratorTypes().getValue();
  if (iteratorTypes.size() != 3)
    return failure();

  return success(vector::isParallelIterator(iteratorTypes[0]) &&
                 vector::isParallelIterator(iteratorTypes[1]) &&
                 vector::isReductionIterator(iteratorTypes[2]));
}

/// Returns the index of the reduction dimension.
unsigned getReductionDim(vector::ContractionOp contract) {
  AffineMap resultMap = contract.getIndexingMapsArray().back();
  ArrayRef<Attribute> iteratorTypes = contract.getIteratorTypes().getValue();
  for (auto [idx, it] : llvm::enumerate(iteratorTypes)) {
    if (vector::isReductionIterator(it)) {
      return idx;
    }
  }

  // Return the last index as a fallback.
  return resultMap.getNumDims() - 1;
}

unsigned getInnermostParallelDim(vector::ContractionOp contract) {
  AffineMap resultMap = contract.getIndexingMapsArray().back();
  return resultMap.getDimPosition(resultMap.getNumResults() - 1);
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::ContractionOp op,
                                              bool targetSupportsDotProd) {
  // Find the contract dimension to unroll. This depends on whether we use the
  // outer product or inner product lowering. Outer product is the default
  // strategy.
  bool lowerToInnerProd =
      targetSupportsDotProd && succeeded(detectI8ToI32Matmul(op));
  unsigned unrollDim =
      lowerToInnerProd ? getReductionDim(op) : getInnermostParallelDim(op);
  auto iteratorTypes = op.getIteratorTypes().getValue();
  SmallVector<int64_t> nativeSize(iteratorTypes.size(), 1);
  SmallVector<int64_t> bounds;
  op.getIterationBounds(bounds);
  nativeSize[unrollDim] = getComputeVectorSize(bounds[unrollDim]);
  return nativeSize;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::MultiDimReductionOp op) {
  // Unroll all reduction dimensions by size 1 for vector.multi_reduction.
  VectorType srcVectorType = op.getSourceVectorType();
  auto nativeSize = llvm::to_vector(srcVectorType.getShape());
  ArrayRef<int64_t> dims = op.getReductionDims();
  for (const int64_t dim : dims) {
    nativeSize[dim] = 1;
  }
  return nativeSize;
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

SmallVector<int64_t> getNativeVectorShapeImpl(vector::GatherOp op) {
  VectorType vectorType = op.getVectorType();
  SmallVector<int64_t> nativeSize(vectorType.getRank(), 1);
  nativeSize.back() = getComputeVectorSize(vectorType.getShape().back());
  return nativeSize;
}

std::optional<SmallVector<int64_t>>
getNativeVectorShape(Operation *op, bool targetSupportsDotProd) {
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = getComputeVectorSize(vecType.getShape().back());
      return nativeSize;
    }
  }

  return TypeSwitch<Operation *, std::optional<SmallVector<int64_t>>>(op)
      .Case<VectorTransferOpInterface, vector::MultiDimReductionOp,
            vector::ReductionOp, vector::TransposeOp, vector::GatherOp>(
          [](auto typedOp) { return getNativeVectorShapeImpl(typedOp); })
      .Case<vector::ContractionOp>([=](auto contract) {
        return getNativeVectorShapeImpl(contract, targetSupportsDotProd);
      })
      .Default([](Operation *) { return std::nullopt; });
}

/// Adds patterns to unroll vector ops to SPIR-V native vector size.
void populateVectorUnrollPatterns(RewritePatternSet &patterns,
                                  bool targetSupportsDotProd) {
  auto options = vector::UnrollVectorOptions().setNativeShapeFn(
      [=](auto op) { return getNativeVectorShape(op, targetSupportsDotProd); });
  vector::populateVectorUnrollPatterns(patterns, options);
}

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

struct LLVMGPUVectorLoweringPass final
    : impl::LLVMGPUVectorLoweringPassBase<LLVMGPUVectorLoweringPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    {
      // Lower high level vector operations like contract or multidim reduce ops
      // to lower level vector ops.
      RewritePatternSet contractLoweringPatterns(funcOp.getContext());
      auto options =
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct);
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          contractLoweringPatterns);
      vector::TransposeOp::getCanonicalizationPatterns(contractLoweringPatterns,
                                                       funcOp.getContext());
      vector::populateVectorBroadcastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorContractLoweringPatterns(
          contractLoweringPatterns, options.vectorContractLowering);
      contractLoweringPatterns.add<PromoteContractOperands>(
          funcOp->getContext());
      vector::populateVectorMaskOpLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorShapeCastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorMultiReductionLoweringPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerParallel);
      if (failed(applyPatternsGreedily(funcOp,
                                       std::move(contractLoweringPatterns)))) {
        return signalPassFailure();
      }
    }
    {

      RewritePatternSet vectorToLoopsPatterns(&getContext());
      VectorTransferToSCFOptions vectorToSCFOptions;
      vectorToSCFOptions.enableFullUnroll();
      populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                            vectorToSCFOptions);
      memref::populateFoldMemRefAliasOpPatterns(vectorToLoopsPatterns);
      vector::populateVectorTransferLoweringPatterns(vectorToLoopsPatterns);
      if (failed(applyPatternsGreedily(funcOp,
                                       std::move(vectorToLoopsPatterns)))) {
        return signalPassFailure();
      }
    }

    // Then unroll vectors to native vector size. We try to use 128-bit
    // vectors for memory access and 4/2/1 vector sizes for computation.
    {
      RewritePatternSet patterns(&getContext());
      populateVectorUnrollPatterns(patterns, false);
      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Next run canonicalization to cast away leading size-1 dimensions. They
    // can be generated from vector unrolling and generally cause issues to
    // cancel corresponding read/write or insert/extract op pairs. This also
    // need to happen before hoisting, where we would make certain vectors loop
    // carried. Once that's done, it's hard to handle the leading size-1
    // dimensions across regions.
    {
      MLIRContext *context = funcOp.getContext();
      RewritePatternSet patterns(context);

      // We need to pull in casting way leading one dims to allow cancelling
      // some read/write ops.
      vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);

      // We may have vector.insert_strided_slice inserting 1-D native vectors
      // into n-D larger vectors with the above. Break that down too. This is a
      // companion transformation of unrolling.
      vector::populateVectorInsertExtractStridedSliceDecompositionPatterns(
          patterns);
      vector::ExtractOp::getCanonicalizationPatterns(patterns, context);

      // Trimming leading unit dims may generate broadcast/shape_cast ops. Clean
      // them up.
      vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
      vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);

      vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
      vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);

      populateVectorTransferTensorSliceTransforms(patterns);

      if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
