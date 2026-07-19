set(TARGET sd-session)

add_executable(${TARGET}
    ../common/cli_params.cpp
    ../common/common.cpp
    ../common/log.cpp
    ../common/media_io.cpp
    generate.cpp
    instrument.cpp
    main.cpp
    serialize.cpp
    session.cpp
    session_params.cpp
    tokenize.cpp
)
if(APPLE)
    sd_set_macos_rpaths(${TARGET})
endif()
target_include_directories(${TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/src"
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
if(SD_WEBP)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBP)
    target_link_libraries(${TARGET} PRIVATE webp libwebpmux)
endif()
if(SD_WEBM)
    target_compile_definitions(${TARGET} PRIVATE SD_USE_WEBM)
    target_link_libraries(${TARGET} PRIVATE webm)
endif()
target_compile_features(${TARGET} PUBLIC c_std_11 cxx_std_17)

add_executable(sd-session-tests tests/test_tokenize.cpp tokenize.cpp)
target_include_directories(sd-session-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(sd-session-tests-instrument tests/test_instrument.cpp instrument.cpp)
target_include_directories(sd-session-tests-instrument PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(sd-session-tests-instrument PRIVATE stable-diffusion ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(sd-session-tests-instrument PUBLIC c_std_11 cxx_std_17)

add_executable(sd-session-tests-params
    ../common/cli_params.cpp
    ../common/common.cpp
    ../common/log.cpp
    ../common/media_io.cpp
    tests/test_apply_flags.cpp
    session_params.cpp
)
target_include_directories(sd-session-tests-params PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(sd-session-tests-params PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
if(SD_WEBP)
    target_compile_definitions(sd-session-tests-params PRIVATE SD_USE_WEBP)
    target_link_libraries(sd-session-tests-params PRIVATE webp libwebpmux)
endif()
if(SD_WEBM)
    target_compile_definitions(sd-session-tests-params PRIVATE SD_USE_WEBM)
    target_link_libraries(sd-session-tests-params PRIVATE webm)
endif()
target_compile_features(sd-session-tests-params PUBLIC c_std_11 cxx_std_17)

add_executable(sd-session-tests-serialize
    ../common/cli_params.cpp
    ../common/common.cpp
    ../common/log.cpp
    ../common/media_io.cpp
    serialize.cpp
    tests/test_serialize.cpp
)
target_include_directories(sd-session-tests-serialize PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${PROJECT_SOURCE_DIR}/src"
)
target_link_libraries(sd-session-tests-serialize PRIVATE stable-diffusion zip ${CMAKE_THREAD_LIBS_INIT})
if(SD_WEBP)
    target_compile_definitions(sd-session-tests-serialize PRIVATE SD_USE_WEBP)
    target_link_libraries(sd-session-tests-serialize PRIVATE webp libwebpmux)
endif()
if(SD_WEBM)
    target_compile_definitions(sd-session-tests-serialize PRIVATE SD_USE_WEBM)
    target_link_libraries(sd-session-tests-serialize PRIVATE webm)
endif()
target_compile_features(sd-session-tests-serialize PUBLIC c_std_11 cxx_std_17)
