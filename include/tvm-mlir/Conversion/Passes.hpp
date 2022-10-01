#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createRelayToAffine();

std::unique_ptr<Pass> createAffineToLLVM();

#define GEN_PASS_REGISTRATION
#include "tvm-mlir/Conversion/Passes.h.inc"

}  // namespace mlir
