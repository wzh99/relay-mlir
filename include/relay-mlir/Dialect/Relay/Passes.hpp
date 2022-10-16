#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace relay {

std::unique_ptr<Pass> createShapeInference();

std::unique_ptr<Pass> createOpFusion();

#define GEN_PASS_REGISTRATION
#include "relay-mlir/Dialect/Relay/Passes.h.inc"

}  // namespace relay
}  // namespace mlir
