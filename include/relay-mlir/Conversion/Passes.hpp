#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createRelayToAffine();

std::unique_ptr<Pass> createAffineToSCF();

std::unique_ptr<Pass> createSCFToLLVM();

#define GEN_PASS_REGISTRATION
#include "relay-mlir/Conversion/Passes.h.inc"

}  // namespace mlir
