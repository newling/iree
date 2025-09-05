

func.func @foo(%arg0: tensor<4096x4096xf16>, %arg1: tensor<4096x4096xf16>, %arg2 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
  %out = linalg.matmul ins(
    %arg0, %arg1 : tensor<4096x4096xf16>, tensor<4096x4096xf16>
  ) outs(
    %arg2 : tensor<4096x4096xf32>
  ) -> tensor<4096x4096xf32>
  return %out : tensor<4096x4096xf32>
}
