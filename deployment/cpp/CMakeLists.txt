cmake_minimum_required(VERSION 3.18)
project(LoMa VERSION 1.0.0 LANGUAGES CXX)

option(LOMA_BUILD_EXAMPLES "Build the match/benchmark example apps" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- OpenCV ----
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ---- ONNX Runtime ----
# Point CMake at your install with -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime
# (on JetPack it is usually /usr/lib/aarch64-linux-gnu + /usr/include).
set(ONNXRUNTIME_ROOT_DIR "" CACHE PATH "Root of an ONNX Runtime install")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT_DIR}
  PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
                onnxruntime/core/session
  PATHS /usr/include /usr/local/include /usr/include/onnxruntime
        /opt/onnxruntime/include)
find_library(ONNXRUNTIME_LIB onnxruntime
  HINTS ${ONNXRUNTIME_ROOT_DIR}
  PATH_SUFFIXES lib lib64
  PATHS /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib /opt/onnxruntime/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR
    "ONNX Runtime not found. Set -DONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime "
    "(must contain include/onnxruntime_cxx_api.h and lib/libonnxruntime.so).")
endif()
message(STATUS "ONNX Runtime headers: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

# ---- library ----
add_library(loma src/loma.cpp)
add_library(LoMa::loma ALIAS loma)
target_include_directories(loma PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_include_directories(loma PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(loma PUBLIC ${OpenCV_LIBS} PRIVATE ${ONNXRUNTIME_LIB})
set_target_properties(loma PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)

# ---- examples ----
if(LOMA_BUILD_EXAMPLES)
  add_executable(loma_match examples/match_example.cpp)
  target_link_libraries(loma_match PRIVATE loma ${OpenCV_LIBS})
  add_executable(loma_bench examples/benchmark.cpp)
  target_link_libraries(loma_bench PRIVATE loma ${OpenCV_LIBS})
endif()

# ---- install + find_package(LoMa) support ----
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
install(TARGETS loma EXPORT LoMaTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT LoMaTargets NAMESPACE LoMa::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LoMa)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/LoMaConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/LoMaConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LoMa)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/LoMaConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/LoMaConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/LoMaConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LoMa)
