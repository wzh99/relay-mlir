add_mlir_library(RelayMLIRTransforms
    OptimizeAffine.cpp

    DEPENDS
    RelayMLIRTransformsPassIncGen

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRDialectUtils
    MLIRRelayDialect
    MLIRAffineDialect
    MLIRArithmeticDialect
    MLIRMemRefDialect
    MLIRAffineTransforms
)
