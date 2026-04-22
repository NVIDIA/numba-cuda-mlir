configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-nvvm70 "Running nvvm70 regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS nvvm70-translate
)

add_custom_target(check-nvvm70-matrix
  COMMAND ${CMAKE_COMMAND} -E env
    NVVM70_BUILD_DIR=${CMAKE_BINARY_DIR}
    bash ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/run-test-matrix.sh
  DEPENDS nvvm70-translate
  COMMENT "Running nvvm70 CUDA × SM test matrix"
  WORKING_DIRECTORY ${NVVM70_SOURCE_DIR}
)
