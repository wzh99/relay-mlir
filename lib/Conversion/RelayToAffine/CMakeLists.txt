add_mlir_conversion_library(MLIRRelayToAffine
    RelayToAffine.cpp

    DEPENDS
    TVMMLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRArithmeticDialect
    MLIRDialectUtils
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRTransforms
    MLIRRelayDialect
    MLIRSupport
    )