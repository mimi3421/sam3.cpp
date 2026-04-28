# Example executables for sam3.cpp

# Quantization tool (no GUI dependency)
add_executable(sam3_quantize quantize.cpp)
target_link_libraries(sam3_quantize PRIVATE ggml)

add_executable(sam3_benchmark benchmark.cpp)
target_link_libraries(sam3_benchmark PRIVATE sam3)

add_executable(sam3_profile_edgetam profile_edgetam.cpp)
target_link_libraries(sam3_profile_edgetam PRIVATE sam3)

# SDL2 + ImGui are optional — build only if SDL2 is found.
find_package(SDL2 QUIET)

if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    if(EXISTS "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
        message(STATUS "Found vcpkg at C:/vcpkg")
    endif()
endif()

if(WIN32 AND DEFINED FFMPEG_ROOT)
    set(FFMPEG_INCLUDE_DIRS "${FFMPEG_ROOT}/include")
    set(FFMPEG_LIBRARIES_DIR "${FFMPEG_ROOT}/lib")
    set(FFMPEG_LIBRARIES "${FFMPEG_LIBRARIES_DIR}/avcodec.lib" "${FFMPEG_LIBRARIES_DIR}/avformat.lib" "${FFMPEG_LIBRARIES_DIR}/avutil.lib" "${FFMPEG_LIBRARIES_DIR}/swscale.lib")
    message(STATUS "Using manual FFmpeg path: ${FFMPEG_ROOT}")
else()
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil libswscale)
        message(STATUS "Found FFmpeg via pkg-config")
    else()
        find_package(FFmpeg REQUIRED QUIET)
        if(FFmpeg_FOUND)
            set(FFMPEG_INCLUDE_DIRS ${FFmpeg_INCLUDE_DIRS})
            set(FFMPEG_LIBRARIES ${FFmpeg_LIBRARIES})
            message(STATUS "Found FFmpeg via find_package")
        else()
            message(FATAL_ERROR "FFmpeg not found. Please install FFmpeg or set FFMPEG_ROOT environment variable.")
        endif()
    endif()
endif()

include_directories(
    ${FFMPEG_INCLUDE_DIRS}
)


if(SDL2_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third-party/CMakeLists.txt")
    add_subdirectory(third-party)

    add_executable(sam3_image main_image.cpp)
    target_link_libraries(sam3_image PRIVATE sam3 imgui-sdl2 SDL2::SDL2)

    #add_executable(sam3_video main_video.cpp)
    #target_link_libraries(sam3_video PRIVATE sam3 imgui-sdl2 SDL2::SDL2)
	
    add_executable(sam3_video_ffmpeg sam3_video_ffmpeg.cpp)
    target_link_libraries(sam3_video_ffmpeg PRIVATE sam3 imgui-sdl2 SDL2::SDL2)
	target_include_directories(sam3_video_ffmpeg PUBLIC
		${FFMPEG_INCLUDE_DIRS}
	)
    target_link_libraries(sam3_video_ffmpeg PUBLIC ${FFMPEG_LIBRARIES})
else()
    message(STATUS "SDL2 not found or third-party/ missing — skipping GUI examples")
endif()
