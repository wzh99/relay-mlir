#ifndef MLIR_RELAY_TRANSFORMS_PASSES
#define MLIR_RELAY_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ShapeInference : Pass<"relay-shape-inference", "func::FuncOp"> {
    let summary = "Shape inference.";
    let description = [{
    Check whether operator attributes and input tensor shapes satisfy 
    constraints of operators, and infer shapes of output tensors. 
    }];

    let constructor = "createShapeInference()";
    let dependentDialects = [
        "func::FuncDialect",
        "relay::RelayDialect"
    ];
}

def OpFusion : Pass<"relay-op-fusion", "func::FuncOp"> {
    let summary = "Operator fusion.";
    let description = [{
    Fuse operators in the main function to a larger function according to some 
    rules.
    }];

    let constructor = "createOpFusion()";
    let dependentDialects = [
        "func::FuncDialect",
        "relay::RelayDialect"
    ];
}

#endif // MLIR_RELAY_TRANSFORMS_PASSES