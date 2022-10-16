#pragma once

#include "fmt/core.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {

/// Fatal error
template <class S, class... Args>
[[noreturn]] inline void Fatal(const S &format, Args &&...args) {
    auto msg = fmt::format(format, std::forward<Args>(args)...);
    llvm::report_fatal_error(llvm::StringRef(msg));
}

/// Error
template <class S, class... Args>
inline void Error(const S &format, Args &&...args) {
    llvm::errs() << fmt::format(format, std::forward<Args>(args)...) << '\n';
}

/// Error
template <class S, class... Args>
inline void Debug(const S &format, Args &&...args) {
    llvm::dbgs() << fmt::format(format, std::forward<Args>(args)...) << '\n';
}

}  // namespace mlir
