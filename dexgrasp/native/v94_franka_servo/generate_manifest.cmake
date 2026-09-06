foreach(required IN ITEMS MANIFEST_TEMPLATE MANIFEST_OUTPUT BINARY_SHA256
                          PRODUCER_BUILD_SHA256 LIBFRANKA_PATH
                          LIBFRANKA_SHA256 LIBFRANKA_SOURCE_COMMIT)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
configure_file("${MANIFEST_TEMPLATE}" "${MANIFEST_OUTPUT}" @ONLY)
