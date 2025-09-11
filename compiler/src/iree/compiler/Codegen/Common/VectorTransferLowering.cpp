// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-vector-transfer-lowering"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VECTORTRANSFERLOWERINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {
class VectorTransferLoweringPass
    : public impl::VectorTransferLoweringPassBase<VectorTransferLoweringPass> {
public:
  using impl::VectorTransferLoweringPassBase<
      VectorTransferLoweringPass>::VectorTransferLoweringPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }
  void runOnOperation() override;
};

void VectorTransferLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto funcOp = getOperation();

  // Try to simplify.
  // Find a transfer read where there result is of type (8xbf16) for now, like:

  SmallVector<vector::TransferReadOp> xferReads;
  funcOp->walk([&](vector::TransferReadOp xferRead) {
    VectorType type = xferRead.getVectorType();
    auto elmType = type.getElementType();
    auto nElms = type.getNumElements();
    if (elmType.isBF16() && nElms == 8) {
      xferReads.push_back(xferRead);
    }
  });

  mlir::IRRewriter rewriter(ctx);
  for (auto xferRead : xferReads) {
    llvm::errs() << "Processing transfer read " << xferRead << "\n";
    auto loc = xferRead.getLoc();
    rewriter.setInsertionPoint(xferRead);
    auto mask = xferRead.getMask();
    auto mask0 = vector::ExtractOp::create(rewriter, loc, mask, 0);
    auto ifOp =
        scf::IfOp::create(rewriter, loc, xferRead.getVectorType(), mask0, true);

    // Then
    auto &thenRegion = ifOp.getThenRegion();
    rewriter.setInsertionPointToStart(&thenRegion.back());
    auto straightNoChaser = vector::TransferReadOp::create(
        rewriter, loc, xferRead.getType(), xferRead.getBase(),
        xferRead.getIndices(), std::optional<Value>{},
        xferRead.getPermutationMap(), xferRead.getInBoundsValues());
    //  (void)straightNoChaser;

    scf::YieldOp::create(rewriter, loc, straightNoChaser.getVector());

    // Else
    auto &elseRegion = ifOp.getElseRegion();
    rewriter.setInsertionPointToStart(&elseRegion.back());
    // elseRegion.back().getTerminator()->setOperand(0, xferRead.getMask());
    auto constantValue =
        vector::BroadcastOp::create(rewriter, loc, xferRead.getVectorType(),
                                    xferRead.getPadding())
            .getResult();
    scf::YieldOp::create(rewriter, loc, constantValue);

    llvm::errs() << "The new scf if op is " << ifOp << "\n";
    rewriter.replaceOp(xferRead, ifOp.getResult(0));
  }

  RewritePatternSet patterns(ctx);
  // Explicitly materialize the mask on transfer_read/transfer_write.
  // Assume we don't have 4 GB vectors.
  vector::populateVectorMaskMaterializationPatterns(
      patterns, /*force32BitVectorIndices=*/true);
  vector::populateVectorTransferLoweringPatterns(patterns,
                                                 /*maxTransferRank=*/1);
  auto vectorTransferToSCFOptions =
      VectorTransferToSCFOptions().enableFullUnroll();
  if (enableScalableLowerings) {
    vectorTransferToSCFOptions.enableLowerScalable();
  }

  populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));

  // Hoist scf.if above scf.for. For example
  //
  //    %20 = scf.for %arg0 = %c0 to %c8 step %c1 iter_args(%arg1 = %cst_0) ->
  //    (vector<8xbf16>) { %67 = affine.apply #map(%arg0)[%5, %19] %68 =
  //    vector.extract %18[0] : i1 from vector<8xi1> %69 = scf.if %68 ->
  //    (vector<8xbf16>) {
  //      %70 = memref.load %1[%67, %4#1] : memref<147456x384xbf16,
  //      #amdgpu.address_space<fat_raw_buffer>> %71 = vector.insert %70, %arg1
  //      [%arg0] : bf16 into vector<8xbf16> scf.yield %71 : vector<8xbf16>
  //    } else {
  //      scf.yield %arg1 : vector<8xbf16>
  //    }
  //    scf.yield %69 : vector<8xbf16>
  //  }
}
} // namespace
} // namespace mlir::iree_compiler
