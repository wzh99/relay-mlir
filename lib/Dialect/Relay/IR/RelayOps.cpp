#include "relay-mlir/Dialect/Relay/RelayOps.hpp"

#include "mlir/IR/OpImplementation.h"
#include "relay-mlir/Dialect/Relay/RelayDialect.hpp"
#include "relay-mlir/Support/Common.hpp"

#define GET_OP_CLASSES
#include "relay-mlir/Dialect/Relay/RelayOps.cpp.inc"

namespace mlir {
namespace relay {

LogicalResult ConstantOp::inferReturnTypeComponents(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    auto value = attributes.get("value");
    auto type = value.getType().cast<TensorType>();
    inferredReturnShapes.push_back({type.getShape()});
    return success();
}

LogicalResult ReLUOp::inferReturnTypeComponents(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    inferredReturnShapes.push_back({operands.getShape(0)});
    return success();
}

LogicalResult DenseOp::inferReturnTypeComponents(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    SmallVector<int64_t> dataShape, weightShape;
    operands.getShape(0).getDims(dataShape);
    operands.getShape(1).getDims(weightShape);
    if (dataShape.size() != 2) {
        Error("Expect rank 2 for data tensor, got {}.", dataShape.size());
        return failure();
    }
    if (weightShape.size() != 2) {
        Error("Expect rank 2 for weight tensor, got {}.", weightShape.size());
        return failure();
    }
    if (dataShape[1] != weightShape[1]) {
        Error("Expect data.shape[1] == weight.shape[1], got {} != {}.",
              dataShape[1], weightShape[1]);
        return failure();
    }
    inferredReturnShapes.push_back(
        llvm::makeArrayRef({dataShape[0], weightShape[0]}));
    return success();
}

LogicalResult BiasAddOp::inferReturnTypeComponents(
    MLIRContext *context, llvm::Optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes, RegionRange regions,
    llvm::SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
    SmallVector<int64_t> dataShape, biasShape;
    operands.getShape(0).getDims(dataShape);
    operands.getShape(1).getDims(biasShape);
    if (biasShape.size() != 1) {
        Error("Expect rank 1 for bias tensor, got {}.", biasShape.size());
        return failure();
    }
    auto dataRank = int64_t(dataShape.size());
    auto axis = attributes.get("axis").cast<IntegerAttr>().getSInt();
    if (axis < -dataRank || axis >= dataRank) {
        Error("Expect axis in range [{}, {}), got {}.", -dataRank, dataRank,
              axis);
        return failure();
    }
    if (axis < 0) axis += dataRank;
    if (dataShape[axis] != biasShape[0]) {
        Error("Expect data.shape[axis] == bias.shape[0], got {} != {}.",
              dataShape[axis], biasShape[0]);
        return failure();
    }
    inferredReturnShapes.push_back({dataShape});
    return success();
}

}  // namespace relay
}  // namespace mlir
