add_mlir_dialect_library(MLIRRelayDialect
    RelayDialect.cpp
    RelayOps.cpp

    DEPENDS
    MLIRRelayOpsIncGen

    LINK_LIBS PUBLIC
    RelayMLIRSupport
    MLIRIR
    MLIRInferTypeOpInterface
)
