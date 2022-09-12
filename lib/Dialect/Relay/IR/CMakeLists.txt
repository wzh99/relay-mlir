add_mlir_dialect_library(MLIRRelay
    RelayDialect.cpp
    RelayOps.cpp

    DEPENDS
    MLIRRelayOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)
