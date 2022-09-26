add_mlir_dialect_library(MLIRRelayTransforms
    ShapeInference.cpp
    OpFusion.cpp

    DEPENDS
    MLIRRelayPassIncGen

    LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRPass
    MLIRRelayDialect
    MLIRTransformUtils
    TVMMLIRSupport
)
