#pragma once

#include "mlir/Pass/Pass.h"
#include "RelayToAffine.hpp"

namespace mlir {

#define GEN_PASS_REGISTRATION
#include "tvm-mlir/Conversion/Passes.h.inc"

}  // namespace mlir
