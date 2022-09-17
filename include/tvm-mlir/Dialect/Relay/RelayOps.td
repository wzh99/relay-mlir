#ifndef MLIR_TVM_RELAY_OPS
#define MLIR_TVM_RELAY_OPS

include "tvm-mlir/Dialect/Relay/RelayDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

/// Relay operator
class Relay_Op<string mnemonic, list<Trait> traits> : Op<Relay_Dialect, mnemonic, traits>;

/// Constant
def Relay_ConstantOp : Relay_Op<"constant", 
        [ConstantLike, NoSideEffect, InferTensorType]> {
    let summary = "A constant tensor.";
    let description = [{
    This operator represents a constant tensor value.
    }];

    let arguments = (ins ElementsAttr:$value);
    let results = (outs AnyStaticShapeTensor:$result);
}

/// Unary-elementwise operators
class Relay_UnaryElementwiseOp<string mnemonic, list<Trait> traits, Type ArgType, 
        Type ResultType = ArgType> : Relay_Op<mnemonic, 
        traits # [Elementwise, NoSideEffect, InferTensorType]> {
    let arguments = (ins ArgType:$data);
    let results = (outs ResultType:$result);
}

// nn.relu
def Relay_ReLUOp : Relay_UnaryElementwiseOp<"nn.relu", [], AnyTensor> {
    let summary = "ReLU operator.";
    let description = [{
    Rectified linear unit.
    }];
}

// nn.dense
def Relay_DenseOp : Relay_Op<"nn.dense", 
    [SameOperandsElementType, NoSideEffect, InferTensorType]> {
    let summary = "Dense operator.";
    let description = [{
    Applies a linear transformation Y = X * W^T. 
    }];

    let arguments = (ins AnyTensor:$data, AnyTensor:$weight);
    let results = (outs AnyTensor:$result);
}

// nn.bias_add
def Relay_BiasAddOp : Relay_Op<"nn.bias_add", 
        [SameOperandsElementType, NoSideEffect, InferTensorType]> {
    let summary = "Add-bias operator.";
    let description = [{
    Add 1D bias to the axis of data. 
    }];

    let arguments = (ins AnyTensor:$data, AnyTensor:$bias, SI64Attr:$axis);
    let results = (outs AnyTensor:$result);
}

#endif
