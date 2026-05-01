configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-llvm70 "Running llvm70 regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS llvm70-translate
)

add_custom_target(check-llvm70-matrix
  COMMAND ${CMAKE_COMMAND} -E env
    LLVM70_BUILD_DIR=${CMAKE_BINARY_DIR}
    bash ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/run-test-matrix.sh
  DEPENDS llvm70-translate
  COMMENT "Running llvm70 CUDA × SM test matrix"
  WORKING_DIRECTORY ${LLVM70_SOURCE_DIR}
)
