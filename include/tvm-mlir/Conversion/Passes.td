#ifndef TVM_MLIR_CONVERSION_PASSES
#define TVM_MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def RelayToAffine : Pass<"relay-to-affine", "ModuleOp"> {
    let summary = "Lower Relay to Affine.";
    let description = [{
    Pass that converts Relay operations to the implementation with the low-level 
    operations in Affine dialect.
    }];

    let constructor = "createRelayToAffine()";
    let dependentDialects = [
        "func::FuncDialect",
        "relay::RelayDialect",
        "AffineDialect",
        "memref::MemRefDialect",
        "arith::ArithmeticDialect"
    ];
}

#endif // TVM_MLIR_CONVERSION_PASSES
