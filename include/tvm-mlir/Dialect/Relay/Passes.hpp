#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace relay {

std::unique_ptr<Pass> createShapeInference();

#define GEN_PASS_REGISTRATION
#include "tvm-mlir/Dialect/Relay/Passes.h.inc"

}  // namespace relay
}  // namespace mlir
