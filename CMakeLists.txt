cmake_minimum_required(VERSION 3.0)
project(cis565_path_tracer)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/cmake")

# Set up include and lib paths
set(EXTERNAL "external")
include_directories("${EXTERNAL}/include")
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/osx")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/linux" "/usr/lib64")
elseif(WIN32)
    set(EXTERNAL_LIB_PATH "${EXTERNAL}/lib/win")
endif()
link_directories(${EXTERNAL_LIB_PATH})
list(APPEND CMAKE_LIBRARY_PATH "${EXTERNAL_LIB_PATH}")

# Configure GLFW
set(GLFW_ROOT_DIR "${EXTERNAL}/glfw-3.2.1")
# We don't need documents, tests, etc..
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${GLFW_ROOT_DIR})
include_directories("${GLFW_ROOT_DIR}/include")
SET(GLFW_LIBRARY glfw)

# Find up and set up core dependency libs
set(GLEW_INCLUDE_DIR "${EXTERNAL}/include")
set(GLEW_LIBRARY_DIR "${CMAKE_LIBRARY_PATH}")
add_definitions(-DGLEW_STATIC)
find_package(GLEW)

find_package(OpenGL)

set(CORELIBS
    "${GLFW_LIBRARY}"
    "${OPENGL_LIBRARY}"
    "${GLEW_LIBRARY}"
    )

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 11)

# Enable CUDA debug info in debug mode builds
list(APPEND CUDA_NVCC_FLAGS_DEBUG -G -g)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

# OSX-specific hacks/fixes
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    list(APPEND CORELIBS "-framework IOKit")
    list(APPEND CORELIBS "-framework Cocoa")
    list(APPEND CORELIBS "-framework CoreVideo")
endif()

# Linux-specific hacks/fixes
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    find_library(x11 NAMES X11)
    find_library(xrandr NAMES Xrandr)
    find_library(xi NAMES Xi)
    find_library(xxf86vm NAMES Xxf86vm)
    list(APPEND CORELIBS ${x11})
    list(APPEND CORELIBS ${xrandr})
    list(APPEND CORELIBS ${xi})
    list(APPEND CORELIBS ${xxf86vm})
endif()

# Crucial magic for CUDA linking
find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(CUDA_SEPARABLE_COMPILATION ON)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

include_directories(.)
add_subdirectory(stream_compaction)  # DONE: uncomment if using your stream compaction
add_subdirectory(src)

cuda_add_executable(${CMAKE_PROJECT_NAME}
    "src/main.h"
    "src/main.cpp"
    )

target_link_libraries(${CMAKE_PROJECT_NAME}
    src
    stream_compaction  # DONE: uncomment if using your stream compaction
    ${CORELIBS}
    )

add_custom_command(
    TARGET ${CMAKE_PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders
        ${CMAKE_BINARY_DIR}/shaders
    )
