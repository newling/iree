// RUN: iree-opt --iree-codegen-foo-jammer --allow-unregistered-dialect %s | FileCheck %s


// CHECK-LABEL: func @castui(
func.func @castui(%arg0: i32) -> index {
  %0 = arith.index_castui %arg0 : i32 to index
  %1 = flow.dispatch.workload.ordinal %0, 4 : index
  %2 = "foo.op"(%1) : (index) -> (index)
  // CHECK: what is this?
  return %2 : index
}
