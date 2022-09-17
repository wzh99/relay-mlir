#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"

#include "mlir/IR/OpImplementation.h"
#include "tvm-mlir/Dialect/Relay/RelayDialect.hpp"

#define GET_OP_CLASSES
#include "tvm-mlir/Dialect/Relay/RelayOps.cpp.inc"

namespace mlir {
namespace relay {

LogicalResult ConstantOp::inferReturnTypeComponents(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    return success();
}

LogicalResult ReLUOp::inferReturnTypeComponents(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    return success();
}

LogicalResult DenseOp::inferReturnTypeComponents(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    return success();
}

LogicalResult BiasAddOp::inferReturnTypeComponents(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    return success();
}

}  // namespace relay
}  // namespace mlir
