add_mlir_library(TVMMLIRTransforms
    OptimizeAffine.cpp

    DEPENDS
    TVMMLIRTransformsPassIncGen

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRDialectUtils
    MLIRRelayDialect
    MLIRAffineDialect
    MLIRArithmeticDialect
    MLIRMemRefDialect
    MLIRAffineTransforms
)
