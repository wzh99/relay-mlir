#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createRelayToAffine();

}