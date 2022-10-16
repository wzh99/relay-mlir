# Relay-MLIR

## Introduction

This repository contains an MLIR-based toy DL compiler which compiles TVM Relay IR to LLVM IR for x86-64 target. Relay is chosen as the input of this compiler because it has a simple and concise text representation of a DL computation graph. 

This project is intended to:

* Demonstrate how to build a simple DL compiler with MLIR; 
* Implement some key components introduced in the [MLC](https://github.com/mlc-ai/mlc-en) course.

This project is NOT intended to:

* Build a full-fledged DL compiler for research or production;
* Achieve comparable performance with mainstream DL compilers. 

## Dependency

* A C++20 compatible compiler
* LLVM 15.0 with MLIR
* TVM 0.8
* [fmt](https://github.com/fmtlib/fmt)

Since TVM does not provide a CMake package, you may have to manually copy its C++ headers and libraries to the build directory of this project. Besides, fmt remains to be a dependency of this project until [C++20 text formatting library](https://en.cppreference.com/w/cpp/utility/format) is fully supported in mainstream implementations of the C++ standard library. 

## Example

Take this [two-layer MLP](example/mlp.txt) as an example. 

```
#[version = "0.0.5"]
def @main(%x: Tensor[(2, 784), float32], %w1: Tensor[(128, 784), float32], %b1: Tensor[(128), float32], %w2: Tensor[(10, 128), float32], %b2: Tensor[(10), float32]) {
    %0 = nn.dense(%x, %w1, units=None);
    %1 = nn.bias_add(%0, %b1, axis=1);
    %2 = nn.relu(%1);
    %3 = nn.dense(%2, %w2, units=None);
    nn.bias_add(%3, %b2, axis=1)
}
```

Currently I have implemented `Relay -> Affine -> SCF+OpenMP -> LLVM` lowering path. Check [`bin/sample.cpp`](bin/sample.cpp) for source code. Other paths (such as the ones with Linalg) will be implemented later. 

First, the Relay IR shown above is imported to MLIR and the tensor shapes are inferred. 

<details><summary><b>Relay</b> <i>[click to expand]</i></summary>
<div>

```mlir
module {
  func.func @main(%arg0: tensor<2x784xf32>, %arg1: tensor<128x784xf32>, %arg2: tensor<128xf32>, %arg3: tensor<10x128xf32>, %arg4: tensor<10xf32>) -> tensor<2x10xf32> {
    %0 = "relay.nn.dense"(%arg0, %arg1) : (tensor<2x784xf32>, tensor<128x784xf32>) -> tensor<2x128xf32>
    %1 = "relay.nn.bias_add"(%0, %arg2) {axis = 1 : si64} : (tensor<2x128xf32>, tensor<128xf32>) -> tensor<2x128xf32>
    %2 = "relay.nn.relu"(%1) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %3 = "relay.nn.dense"(%2, %arg3) : (tensor<2x128xf32>, tensor<10x128xf32>) -> tensor<2x10xf32>
    %4 = "relay.nn.bias_add"(%3, %arg4) {axis = 1 : si64} : (tensor<2x10xf32>, tensor<10xf32>) -> tensor<2x10xf32>
    return %4 : tensor<2x10xf32>
  }
}
```

</div>
</details>

Next, the operations are fused into groups. A new function is created for each fusion group. 

<details><summary><b>Relay (fused)</b> <i>[click to expand]</i></summary>
<div>

```mlir
module {
  func.func @main(%arg0: tensor<2x784xf32>, %arg1: tensor<128x784xf32>, %arg2: tensor<128xf32>, %arg3: tensor<10x128xf32>, %arg4: tensor<10xf32>) -> tensor<2x10xf32> {
    %0 = call @fused_0(%arg0, %arg1, %arg2) : (tensor<2x784xf32>, tensor<128x784xf32>, tensor<128xf32>) -> tensor<2x128xf32>
    %1 = call @fused_1(%0, %arg3, %arg4) : (tensor<2x128xf32>, tensor<10x128xf32>, tensor<10xf32>) -> tensor<2x10xf32>
    return %1 : tensor<2x10xf32>
  }
  func.func @fused_0(%arg0: tensor<2x784xf32>, %arg1: tensor<128x784xf32>, %arg2: tensor<128xf32>) -> tensor<2x128xf32> attributes {primitive = true} {
    %0 = "relay.nn.dense"(%arg0, %arg1) : (tensor<2x784xf32>, tensor<128x784xf32>) -> tensor<2x128xf32>
    %1 = "relay.nn.bias_add"(%0, %arg2) {axis = 1 : si64} : (tensor<2x128xf32>, tensor<128xf32>) -> tensor<2x128xf32>
    %2 = "relay.nn.relu"(%1) : (tensor<2x128xf32>) -> tensor<2x128xf32>
    return %2 : tensor<2x128xf32>
  }
  func.func @fused_1(%arg0: tensor<2x128xf32>, %arg1: tensor<10x128xf32>, %arg2: tensor<10xf32>) -> tensor<2x10xf32> attributes {primitive = true} {
    %0 = "relay.nn.dense"(%arg0, %arg1) : (tensor<2x128xf32>, tensor<10x128xf32>) -> tensor<2x10xf32>
    %1 = "relay.nn.bias_add"(%0, %arg2) {axis = 1 : si64} : (tensor<2x10xf32>, tensor<10xf32>) -> tensor<2x10xf32>
    return %1 : tensor<2x10xf32>
  }
}
```

</div>
</details>

We can then lower Relay dialect to Affine dialect. Note that the buffers (`memref`s) are allocated and deallocated based on their lifetimes. 

<details><summary><b>Affine</b> <i>[click to expand]</i></summary>
<div>

```mlir
module {
  func.func @main(%arg0: memref<2x784xf32>, %arg1: memref<128x784xf32>, %arg2: memref<128xf32>, %arg3: memref<10x128xf32>, %arg4: memref<10xf32>, %arg5: memref<2x10xf32>) attributes {num_inputs = 5 : i64} {
    %0 = memref.alloc() : memref<2x128xf32>
    call @fused_0_lowered(%arg0, %arg1, %arg2, %0) : (memref<2x784xf32>, memref<128x784xf32>, memref<128xf32>, memref<2x128xf32>) -> ()
    call @fused_1_lowered(%0, %arg3, %arg4, %arg5) : (memref<2x128xf32>, memref<10x128xf32>, memref<10xf32>, memref<2x10xf32>) -> ()
    memref.dealloc %0 : memref<2x128xf32>
    return
  }
  func.func @fused_0_lowered(%arg0: memref<2x784xf32>, %arg1: memref<128x784xf32>, %arg2: memref<128xf32>, %arg3: memref<2x128xf32>) attributes {num_inputs = 3 : i64, primitive = true} {
    %0 = memref.alloc() : memref<2x128xf32>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 128 {
        %cst = arith.constant 0.000000e+00 : f32
        %2 = affine.for %arg6 = 0 to 784 iter_args(%arg7 = %cst) -> (f32) {
          %3 = affine.load %arg0[%arg4, %arg6] : memref<2x784xf32>
          %4 = affine.load %arg1[%arg5, %arg6] : memref<128x784xf32>
          %5 = arith.mulf %3, %4 : f32
          %6 = arith.addf %arg7, %5 : f32
          affine.yield %6 : f32
        }
        affine.store %2, %0[%arg4, %arg5] : memref<2x128xf32>
      }
    }
    %1 = memref.alloc() : memref<2x128xf32>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 128 {
        %2 = affine.load %0[%arg4, %arg5] : memref<2x128xf32>
        %3 = affine.load %arg2[%arg5] : memref<128xf32>
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %1[%arg4, %arg5] : memref<2x128xf32>
      }
    }
    memref.dealloc %0 : memref<2x128xf32>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 128 {
        %2 = affine.load %1[%arg4, %arg5] : memref<2x128xf32>
        %cst = arith.constant 0.000000e+00 : f32
        %3 = arith.maxf %2, %cst : f32
        affine.store %3, %arg3[%arg4, %arg5] : memref<2x128xf32>
      }
    }
    memref.dealloc %1 : memref<2x128xf32>
    return
  }
  func.func @fused_1_lowered(%arg0: memref<2x128xf32>, %arg1: memref<10x128xf32>, %arg2: memref<10xf32>, %arg3: memref<2x10xf32>) attributes {num_inputs = 3 : i64, primitive = true} {
    %0 = memref.alloc() : memref<2x10xf32>
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 10 {
        %cst = arith.constant 0.000000e+00 : f32
        %1 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst) -> (f32) {
          %2 = affine.load %arg0[%arg4, %arg6] : memref<2x128xf32>
          %3 = affine.load %arg1[%arg5, %arg6] : memref<10x128xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = arith.addf %arg7, %4 : f32
          affine.yield %5 : f32
        }
        affine.store %1, %0[%arg4, %arg5] : memref<2x10xf32>
      }
    }
    affine.for %arg4 = 0 to 2 {
      affine.for %arg5 = 0 to 10 {
        %1 = affine.load %0[%arg4, %arg5] : memref<2x10xf32>
        %2 = affine.load %arg2[%arg5] : memref<10xf32>
        %3 = arith.addf %1, %2 : f32
        affine.store %3, %arg3[%arg4, %arg5] : memref<2x10xf32>
      }
    }
    memref.dealloc %0 : memref<2x10xf32>
    return
  }
}
```

</div>
</details>

Several optimizations are performed on affine loops, including loop fusion, scalar replacement, parallelization and vectorization.

<details><summary><b>Affine (optimized)</b> <i>[click to expand]</i></summary>
<div>

```mlir
module {
  func.func @main(%arg0: memref<2x784xf32>, %arg1: memref<128x784xf32>, %arg2: memref<128xf32>, %arg3: memref<10x128xf32>, %arg4: memref<10xf32>, %arg5: memref<2x10xf32>) attributes {num_inputs = 5 : i64} {
    %0 = memref.alloc() : memref<2x128xf32>
    call @fused_0_lowered(%arg0, %arg1, %arg2, %0) : (memref<2x784xf32>, memref<128x784xf32>, memref<128xf32>, memref<2x128xf32>) -> ()
    call @fused_1_lowered(%0, %arg3, %arg4, %arg5) : (memref<2x128xf32>, memref<10x128xf32>, memref<10xf32>, memref<2x10xf32>) -> ()
    memref.dealloc %0 : memref<2x128xf32>
    return
  }
  func.func @fused_0_lowered(%arg0: memref<2x784xf32>, %arg1: memref<128x784xf32>, %arg2: memref<128xf32>, %arg3: memref<2x128xf32>) attributes {num_inputs = 3 : i64, primitive = true} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%arg4, %arg5) = (0, 0) to (2, 128) {
      %cst_0 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %0 = affine.for %arg6 = 0 to 784 step 8 iter_args(%arg7 = %cst_0) -> (vector<8xf32>) {
        %cst_1 = arith.constant 0.000000e+00 : f32
        %5 = vector.transfer_read %arg0[%arg4, %arg6], %cst_1 : memref<2x784xf32>, vector<8xf32>
        %cst_2 = arith.constant 0.000000e+00 : f32
        %6 = vector.transfer_read %arg1[%arg5, %arg6], %cst_2 : memref<128x784xf32>, vector<8xf32>
        %7 = arith.mulf %5, %6 : vector<8xf32>
        %8 = arith.addf %arg7, %7 : vector<8xf32>
        affine.yield %8 : vector<8xf32>
      }
      %1 = vector.reduction <add>, %0 : vector<8xf32> into f32
      %2 = affine.load %arg2[%arg5] : memref<128xf32>
      %3 = arith.addf %1, %2 : f32
      %4 = arith.maxf %3, %cst : f32
      affine.store %4, %arg3[%arg4, %arg5] : memref<2x128xf32>
    }
    return
  }
  func.func @fused_1_lowered(%arg0: memref<2x128xf32>, %arg1: memref<10x128xf32>, %arg2: memref<10xf32>, %arg3: memref<2x10xf32>) attributes {num_inputs = 3 : i64, primitive = true} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.parallel (%arg4, %arg5) = (0, 0) to (2, 10) {
      %cst_0 = arith.constant dense<0.000000e+00> : vector<8xf32>
      %0 = affine.for %arg6 = 0 to 128 step 8 iter_args(%arg7 = %cst_0) -> (vector<8xf32>) {
        %cst_1 = arith.constant 0.000000e+00 : f32
        %4 = vector.transfer_read %arg0[%arg4, %arg6], %cst_1 : memref<2x128xf32>, vector<8xf32>
        %cst_2 = arith.constant 0.000000e+00 : f32
        %5 = vector.transfer_read %arg1[%arg5, %arg6], %cst_2 : memref<10x128xf32>, vector<8xf32>
        %6 = arith.mulf %4, %5 : vector<8xf32>
        %7 = arith.addf %arg7, %6 : vector<8xf32>
        affine.yield %7 : vector<8xf32>
      }
      %1 = vector.reduction <add>, %0 : vector<8xf32> into f32
      %2 = affine.load %arg2[%arg5] : memref<10xf32>
      %3 = arith.addf %1, %2 : f32
      affine.store %3, %arg3[%arg4, %arg5] : memref<2x10xf32>
    }
    return
  }
}
```

</div>
</details>

Then we lower Affine dialect to SCF dialect (normal loops) and OpenMP dialect (parallel loops). 


<details><summary><b>SCF and OpenMP</b> <i>[click to expand]</i></summary>
<div>

```mlir
module {
  func.func @main(%arg0: memref<2x784xf32>, %arg1: memref<128x784xf32>, %arg2: memref<128xf32>, %arg3: memref<10x128xf32>, %arg4: memref<10xf32>, %arg5: memref<2x10xf32>) attributes {num_inputs = 5 : i64} {
    %0 = memref.alloc() : memref<2x128xf32>
    call @fused_0_lowered(%arg0, %arg1, %arg2, %0) : (memref<2x784xf32>, memref<128x784xf32>, memref<128xf32>, memref<2x128xf32>) -> ()
    call @fused_1_lowered(%0, %arg3, %arg4, %arg5) : (memref<2x128xf32>, memref<10x128xf32>, memref<10xf32>, memref<2x10xf32>) -> ()
    memref.dealloc %0 : memref<2x128xf32>
    return
  }
  func.func @fused_0_lowered(%arg0: memref<2x784xf32>, %arg1: memref<128x784xf32>, %arg2: memref<128xf32>, %arg3: memref<2x128xf32>) attributes {num_inputs = 3 : i64, primitive = true} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c0_0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %0 = llvm.mlir.constant(1 : i64) : i64
    omp.parallel   {
      omp.wsloop   for  (%arg4, %arg5) : index = (%c0, %c0_0) to (%c2, %c128) step (%c1, %c1_1) {
        memref.alloca_scope  {
          %cst_2 = arith.constant dense<0.000000e+00> : vector<8xf32>
          %c0_3 = arith.constant 0 : index
          %c784 = arith.constant 784 : index
          %c8 = arith.constant 8 : index
          %1 = scf.for %arg6 = %c0_3 to %c784 step %c8 iter_args(%arg7 = %cst_2) -> (vector<8xf32>) {
            %cst_4 = arith.constant 0.000000e+00 : f32
            %6 = vector.transfer_read %arg0[%arg4, %arg6], %cst_4 : memref<2x784xf32>, vector<8xf32>
            %cst_5 = arith.constant 0.000000e+00 : f32
            %7 = vector.transfer_read %arg1[%arg5, %arg6], %cst_5 : memref<128x784xf32>, vector<8xf32>
            %8 = arith.mulf %6, %7 : vector<8xf32>
            %9 = arith.addf %arg7, %8 : vector<8xf32>
            scf.yield %9 : vector<8xf32>
          }
          %2 = vector.reduction <add>, %1 : vector<8xf32> into f32
          %3 = memref.load %arg2[%arg5] : memref<128xf32>
          %4 = arith.addf %2, %3 : f32
          %5 = arith.maxf %4, %cst : f32
          memref.store %5, %arg3[%arg4, %arg5] : memref<2x128xf32>
        }
        omp.yield
      }
      omp.terminator
    }
    return
  }
  func.func @fused_1_lowered(%arg0: memref<2x128xf32>, %arg1: memref<10x128xf32>, %arg2: memref<10xf32>, %arg3: memref<2x10xf32>) attributes {num_inputs = 3 : i64, primitive = true} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c0_0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %0 = llvm.mlir.constant(1 : i64) : i64
    omp.parallel   {
      omp.wsloop   for  (%arg4, %arg5) : index = (%c0, %c0_0) to (%c2, %c10) step (%c1, %c1_1) {
        memref.alloca_scope  {
          %cst_2 = arith.constant dense<0.000000e+00> : vector<8xf32>
          %c0_3 = arith.constant 0 : index
          %c128 = arith.constant 128 : index
          %c8 = arith.constant 8 : index
          %1 = scf.for %arg6 = %c0_3 to %c128 step %c8 iter_args(%arg7 = %cst_2) -> (vector<8xf32>) {
            %cst_4 = arith.constant 0.000000e+00 : f32
            %5 = vector.transfer_read %arg0[%arg4, %arg6], %cst_4 : memref<2x128xf32>, vector<8xf32>
            %cst_5 = arith.constant 0.000000e+00 : f32
            %6 = vector.transfer_read %arg1[%arg5, %arg6], %cst_5 : memref<10x128xf32>, vector<8xf32>
            %7 = arith.mulf %5, %6 : vector<8xf32>
            %8 = arith.addf %arg7, %7 : vector<8xf32>
            scf.yield %8 : vector<8xf32>
          }
          %2 = vector.reduction <add>, %1 : vector<8xf32> into f32
          %3 = memref.load %arg2[%arg5] : memref<10xf32>
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg3[%arg4, %arg5] : memref<2x10xf32>
        }
        omp.yield
      }
      omp.terminator
    }
    return
  }
}
```

</div>
</details>

Finally we lower it to LLVM dialect. The code is omitted here because of its length. You can run the sample program to see for yourself. 

## Reference

* [MLIR Code Documentation](https://mlir.llvm.org/docs/)
* [Machine Learning Compilation](https://mlc.ai/summer22/)
* [Relay Language Reference](https://tvm.apache.org/docs/reference/langref/index.html)
