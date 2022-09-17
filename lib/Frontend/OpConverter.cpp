#include "OpConverter.hpp"

#include "mlir/IR/Builders.h"
#include "tvm-mlir/Dialect/Relay/RelayOps.hpp"
#include "tvm-mlir/Support/Common.hpp"
#include "tvm/ir/op.h"
#include "tvm/relay/attrs/nn.h"

namespace mlir {
namespace relay {

inline static UnrankedTensorType getSameElemTensorType(const Value &value) {
    return UnrankedTensorType::get(
        value.getType().cast<TensorType>().getElementType());
}

#define CVT_FUNC(Op)                                              \
    static OpState cvt##Op(const std::vector<Value> &operands,    \
                           const tvm::Attrs &attrs, Location loc, \
                           OpBuilder &builder)

template <class O, int... argIndices>
inline CVT_FUNC(SameElemTypeNoAttr) {
    return builder.create<O>(loc, getSameElemTensorType(operands[0]),
                             operands[argIndices]...);
}

#define ONE_ARG 0
#define TWO_ARG 0, 1

CVT_FUNC(BiasAdd) {
    auto biasAddAttrs = attrs.as<tvm::relay::BiasAddAttrs>();
    return builder.create<BiasAddOp>(loc, getSameElemTensorType(operands[0]),
                                     operands[0], operands[1],
                                     biasAddAttrs->axis);
}

static std::unordered_map<tvm::String,
                          OpState (*)(const std::vector<Value> &operands,
                                      const tvm::Attrs &attrs, Location loc,
                                      OpBuilder &builder)>
    cvtFuncs{{"nn.relu", cvtSameElemTypeNoAttr<ReLUOp, ONE_ARG>},
             {"nn.dense", cvtSameElemTypeNoAttr<DenseOp, TWO_ARG>},
             {"nn.bias_add", cvtBiasAdd}};

OpState ConvertRelayOp(const tvm::String &name,
                       const std::vector<Value> &operands,
                       const tvm::Attrs &attrs, Location loc,
                       OpBuilder &builder) {
    if (cvtFuncs.count(name))
        return cvtFuncs[name](operands, attrs, loc, builder);
    else
        FatalError("Operator `{}` is not supported.", name.c_str());
}

}  // namespace relay
}  // namespace mlir