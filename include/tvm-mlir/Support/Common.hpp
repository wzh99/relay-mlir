#pragma once

#include "fmt/core.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace relay {

/// Fatal error
template <class S, class... Args>
[[noreturn]] inline void FatalError(const S &format, Args &&...args) {
    auto msg = fmt::format(format, std::forward<Args>(args)...);
    llvm::report_fatal_error(llvm::StringRef(msg));
}

}  // namespace relay
}  // namespace mlir
