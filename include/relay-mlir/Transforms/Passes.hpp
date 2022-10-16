#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createOptimizeAffine();

#define GEN_PASS_REGISTRATION
#include "relay-mlir/Transforms/Passes.h.inc"

}  // namespace mlir
