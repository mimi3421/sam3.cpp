cmake_minimum_required(VERSION 3.14)
project(sam3.cpp VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ggml options — enable Metal on Apple
option(SAM3_METAL "Enable Metal backend" ON)
if(APPLE AND SAM3_METAL)
    set(GGML_METAL        ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

option(SAM3_VULKAN "Enable Vulkan backend" OFF)
if(SAM3_VULKAN)
    find_package(Vulkan REQUIRED)
    set(GGML_VULKAN        ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# VideoFrameExtractor static library
find_package(FFMPEG REQUIRED)
if(WIN32 AND FFMPEG_FOUND)
    #set(FFMPEG_INCLUDE_DIRS "${FFMPEG_ROOT}/include")
    #set(FFMPEG_LIBRARIES_DIR "${FFMPEG_ROOT}/lib")
    #set(FFMPEG_LIBRARIES "${FFMPEG_LIBRARIES_DIR}/avcodec.lib" "${FFMPEG_LIBRARIES_DIR}/avformat.lib" "${FFMPEG_LIBRARIES_DIR}/avutil.lib" "${FFMPEG_LIBRARIES_DIR}/swscale.lib")
    #message(STATUS "Using manual FFmpeg path: ${FFMPEG_ROOT}")

	include_directories(${FFMPEG_INCLUDE_DIRS})

	add_library(VideoFrameExtractor VideoFrameExtractor.cpp)

	target_include_directories(VideoFrameExtractor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${FFMPEG_INCLUDE_DIRS})

	target_link_libraries(VideoFrameExtractor PUBLIC ${FFMPEG_LIBRARIES})
	target_link_directories(VideoFrameExtractor PUBLIC ${FFMPEG_LIBRARY_DIRS})
	
endif()


# sam3 static library
add_library(sam3 STATIC sam3.cpp sam3.h)
target_include_directories(sam3 PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
)

if(SAM3_VULKAN)
    target_link_libraries(sam3 PUBLIC Vulkan::Vulkan)
endif()
target_link_libraries(sam3 PUBLIC ggml)
target_link_libraries(sam3 PUBLIC VideoFrameExtractor)
target_compile_features(sam3 PUBLIC cxx_std_14)


# Examples (only when top-level)
option(SAM3_BUILD_EXAMPLES "Build example executables" ON)
if(SAM3_BUILD_EXAMPLES AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    add_subdirectory(examples)
endif()

# Tests
option(SAM3_BUILD_TESTS "Build test executables" OFF)
if(SAM3_BUILD_TESTS)
    add_subdirectory(tests)
endif()
