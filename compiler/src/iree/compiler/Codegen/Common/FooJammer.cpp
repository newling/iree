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

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FOOJAMMERPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct FooJammerPass final : impl::FooJammerPassBase<FooJammerPass> {
  using impl::FooJammerPassBase<FooJammerPass>::FooJammerPassBase;
  void runOnOperation() override;
};

} // namespace

void FooJammerPass::runOnOperation() {
  if (jamFlavor == 700) {
    llvm::errs() << "Et voila a l'inverse de la magie!" << "\n";
  }
}

} // namespace mlir::iree_compiler
