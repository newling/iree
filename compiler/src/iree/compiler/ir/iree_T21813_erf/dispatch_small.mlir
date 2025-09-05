
// Batch matmul with:
//
// B = 1  <- dynamic, chosen at runtime
// M = 1  <- dynamic, chosen at runtime
// N = 32
// K = 64
//
// Followed by an erf non-linearity.
// This failed with compilation error, see issue
// https://github.com/iree-org/iree/issues/21813

!LHS = tensor<1x1x64xf32>
!LHS_DYN = tensor<?x?x64xf32>

!RHS = tensor<1x64x32xf32>
!RHS_DYN = tensor<?x64x32xf32>

!OUT = tensor<1x1x32xf32>
!OUT_DYN = tensor<?x?x32xf32>

func.func @batch_matmul_B2_M9_N31_K32_followed_by_erf() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_f32 = arith.constant 0.0 : f32

  %lhs = flow.tensor.dynamic_constant dense<0.25> : !LHS -> !LHS_DYN
  %rhs = flow.tensor.dynamic_constant dense<0.125> : !RHS -> !RHS_DYN
  %B = tensor.dim %lhs, %c0 : !LHS_DYN
  %M = tensor.dim %lhs, %c1 : !LHS_DYN

  %empty = tensor.empty(%B, %M) : !OUT_DYN
  %filled = linalg.fill ins(%c0_f32: f32) outs(%empty : !OUT_DYN) -> !OUT_DYN
  %bmm = linalg.batch_matmul ins(%lhs, %rhs : !LHS_DYN, !RHS_DYN)
                                    outs(%filled : !OUT_DYN) -> !OUT_DYN
  %erfed = math.erf %bmm : !OUT_DYN

  // The reduction size K=32, and the values in the LHS and RHS tensors are constants
  // 1/4 and 1/8. So the result of the batched matmul is 32 * (1/4) * (1/8) = 1.0.
  //
  // erf(1.0) =  0.8427007929497149 (check with python `math.erf(1.0)`)
  %expected = flow.tensor.dynamic_constant dense<0.84270079> : !OUT -> !OUT_DYN
  check.expect_almost_eq(%erfed, %expected, atol 1.0e-02) : !OUT_DYN
  return
}
