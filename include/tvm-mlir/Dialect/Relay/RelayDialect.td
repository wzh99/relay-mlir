#ifndef MLIR_TVM_RELAY_DIALECT
#define MLIR_TVM_RELAY_DIALECT

include "mlir/IR/OpBase.td"

def Relay_Dialect : Dialect {
    let name = "relay";
    let cppNamespace = "::mlir::relay";
    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif
