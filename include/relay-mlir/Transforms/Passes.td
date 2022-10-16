#ifndef TVM_MLIR_TRANSFORMS_PASSES
#define TVM_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def OptimizeAffine : Pass<"optimize-affine", "ModuleOp"> {
    let summary = "Optimize affine loops.";
    let description = [{
    Optimize computation expressed in Affine dialect. 
    }];

    let constructor = "createOptimizeAffine()";
}

#endif // TVM_MLIR_TRANSFORMS_PASSES