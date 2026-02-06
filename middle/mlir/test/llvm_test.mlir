// Simple test for LLVM lowering (memref-based, no tensor)
// This file tests: SCF -> CF -> LLVM conversion
//
// Run: ./build/gawee-opt --gawee-to-llvm-from-scf test/llvm_test.mlir

module {
  func.func @simple_loop(%arg0: memref<4xf32>, %arg1: memref<4xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    scf.for %i = %c0 to %c4 step %c1 {
      %val = memref.load %arg0[%i] : memref<4xf32>
      memref.store %val, %arg1[%i] : memref<4xf32>
    }
    return
  }
}
