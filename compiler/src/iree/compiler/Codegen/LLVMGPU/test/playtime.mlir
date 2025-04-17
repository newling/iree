
func.func @original(%arg0: vector<2xf32>) -> vector<32xf32> {
    %0 = vector.shape_cast %arg0 : vector<2xf32> to vector<2x1x1xf32>
    %cst = arith.constant dense<1.000000e+00> : vector<4x2x1x1x4x1xf32>
    %cst_0 = arith.constant dense<7.000000e+00> : vector<4x2x1x1x4x1xf32>
    %cst_1 = arith.constant dense<9.000000e+00> : vector<2x4x1x1x1x4xf32>
    %1 = vector.extract %0[0, 0] : vector<1xf32> from vector<2x1x1xf32>
    %2 = vector.broadcast %1 : vector<1xf32> to vector<4x1xf32>
    %3 = vector.insert %2, %cst [0, 0, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
    %4 = vector.extract %0[1, 0] : vector<1xf32> from vector<2x1x1xf32>
    %5 = vector.broadcast %4 : vector<1xf32> to vector<4x1xf32>
    %6 = vector.insert %5, %3 [0, 1, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
    %7 = vector.insert %2, %6 [1, 0, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
    %8 = vector.insert %5, %7 [1, 1, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
    %9 = vector.insert %2, %8 [2, 0, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
    %10 = vector.insert %5, %9 [2, 1, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
    %11 = vector.insert %2, %10 [3, 0, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
    %12 = vector.insert %5, %11 [3, 1, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
    %13 = arith.divf %cst_0, %12 : vector<4x2x1x1x4x1xf32>
    %14 = vector.transpose %13, [1, 0, 3, 2, 5, 4] : vector<4x2x1x1x4x1xf32> to vector<2x4x1x1x1x4xf32>
    %flat = vector.shape_cast %14 : vector<2x4x1x1x1x4xf32> to vector<32xf32>
    return %flat : vector<32xf32>
  //   %15 = arith.mulf %14, %cst_1 : vector<2x4x1x1x1x4xf32>
  //   %16 = arith.truncf %15 : vector<2x4x1x1x1x4xf32> to vector<2x4x1x1x1x4xf16>
  //   %17 = vector.shape_cast %16 : vector<2x4x1x1x1x4xf16> to vector<32xf16>
    // return %14 : vector<2x4x1x1x1x4xf32>
}


// func.func @test(%arg0: vector<2xf32>) -> vector<32xf16> {
//     %0 = vector.shape_cast %arg0 : vector<2xf32> to vector<2x1x1xf32>
//     %cst = arith.constant dense<1.000000e+00> : vector<4x2x1x1x4x1xf32>
//     %cst_0 = arith.constant dense<7.000000e+00> : vector<2x4x1x1x1x4xf32>
//     %cst_1 = arith.constant dense<9.000000e+00> : vector<2x4x1x1x1x4xf32>
//     %1 = vector.extract %0[0, 0] : vector<1xf32> from vector<2x1x1xf32>
//     %2 = vector.broadcast %1 : vector<1xf32> to vector<4x1xf32>
//     %3 = vector.insert %2, %cst [0, 0, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
//     %4 = vector.extract %0[1, 0] : vector<1xf32> from vector<2x1x1xf32>
//     %5 = vector.broadcast %4 : vector<1xf32> to vector<4x1xf32>
//     %6 = vector.insert %5, %3 [0, 1, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
//     %7 = vector.insert %2, %6 [1, 0, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
//     %8 = vector.insert %5, %7 [1, 1, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
//     %9 = vector.insert %2, %8 [2, 0, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
//     %10 = vector.insert %5, %9 [2, 1, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
//     %11 = vector.insert %2, %10 [3, 0, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
//     %12 = vector.insert %5, %11 [3, 1, 0, 0] : vector<4x1xf32> into vector<4x2x1x1x4x1xf32>
//     %13 = vector.transpose %12, [1, 0, 3, 2, 5, 4] : vector<4x2x1x1x4x1xf32> to vector<2x4x1x1x1x4xf32>
//     %14 = arith.divf %cst_0, %13 : vector<2x4x1x1x1x4xf32>
//     %15 = arith.mulf %14, %cst_1 : vector<2x4x1x1x1x4xf32>
//     %16 = arith.truncf %15 : vector<2x4x1x1x1x4xf32> to vector<2x4x1x1x1x4xf16>
//     %17 = vector.shape_cast %16 : vector<2x4x1x1x1x4xf16> to vector<32xf16>
//     return %17 : vector<32xf16>
// }
