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

#endif // MLIR_RELAY_TRANSFORMS_PASSES