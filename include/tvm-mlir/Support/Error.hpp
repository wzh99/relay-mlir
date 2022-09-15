#pragma once

#include "fmt/core.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace relay {

template <class S, class... Args>
[[noreturn]] inline void FatalError(const S &format, Args &&...args) {
    auto msg = fmt::format(format, args...);
    llvm::report_fatal_error(llvm::StringRef(format));
}

}  // namespace relay
}  // namespace mlir
