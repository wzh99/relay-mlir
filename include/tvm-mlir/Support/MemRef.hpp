#pragma once

#include "mlir/IR/BuiltinTypes.h"

namespace mlir {

class MemRef {
public:
    MemRef(MemRefType type);

    ~MemRef() { std::free(data); }

    void LoadData(const void *src) const { std::memcpy(data, src, this->size); }

    template <class T>
    T *GetDataAs() const { return reinterpret_cast<T *>(data); }

    void PopulateLLJITArgs(SmallVector<void *> &args);

private:
    SmallVector<int64_t> shape, strides;
    size_t size;
    void *data = nullptr;
};

}  // namespace mlir
