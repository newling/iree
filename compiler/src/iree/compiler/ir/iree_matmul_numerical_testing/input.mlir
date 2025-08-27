func.func @foo(%arg0: tensor<32x32xf16>, %arg1: tensor<32x32xf16>, %arg2 : tensor<32x32xf32>) -> tensor<32x32xf32> {
  %out = linalg.matmul ins(%arg0, %arg1 : tensor<32x32xf16>, tensor<32x32xf16>)
                      outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %out : tensor<32x32xf32>
}
