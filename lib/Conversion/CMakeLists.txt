add_mlir_conversion_library(TVMMLIRConversion
    RelayToAffine.cpp
    AffineToSCF.cpp
    SCFToLLVM.cpp

    DEPENDS
    TVMMLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRDialectUtils
    MLIRRelayDialect
    MLIRAffineDialect
    MLIRArithmeticDialect
    MLIRMemRefDialect
    MLIRLLVMDialect
    MLIRArithmeticTransforms
    MLIRAffineToStandard
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMemRefToLLVM
    MLIROpenMPToLLVM
    MLIRVectorToLLVM
    )