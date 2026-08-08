cmake_minimum_required(VERSION 3.20)
project(ResolveTextAnimator VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(OFX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/openfx)
if(NOT EXISTS ${OFX_ROOT}/Support/Library/ofxsCore.cpp)
  message(FATAL_ERROR "OpenFX SDK missing. Run: git clone --depth 1 "
                      "https://github.com/AcademySoftwareFoundation/openfx.git external/openfx")
endif()

# --------------------------------------------------------------- core library
# Host-independent: no OFX headers, so the same code runs in the test harness.
add_library(rta_core STATIC
  src/segmentation.cpp
  src/animator.cpp
  src/compositor.cpp
  src/diagnostics.cpp
)
target_include_directories(rta_core PUBLIC src)
if(MSVC)
  target_compile_options(rta_core PRIVATE /W4 /fp:fast)
endif()

# ------------------------------------------------------------------- OFX plugin
# The Support library is compiled straight into the plugin. Consuming the
# openfx repo's own CMake drags in OpenGL/CUDA options and example plugins.
add_library(TextAnimator SHARED
  src/plugin.cpp
  src/interact/overlay.cpp
  src/interact/curve_widget.cpp
  ${OFX_ROOT}/Support/Library/ofxsCore.cpp
  ${OFX_ROOT}/Support/Library/ofxsImageEffect.cpp
  ${OFX_ROOT}/Support/Library/ofxsInteract.cpp
  ${OFX_ROOT}/Support/Library/ofxsLog.cpp
  ${OFX_ROOT}/Support/Library/ofxsMultiThread.cpp
  ${OFX_ROOT}/Support/Library/ofxsParams.cpp
  ${OFX_ROOT}/Support/Library/ofxsProperty.cpp
  ${OFX_ROOT}/Support/Library/ofxsPropertyValidation.cpp
)
target_link_libraries(TextAnimator PRIVATE rta_core)
target_include_directories(TextAnimator PRIVATE
  ${OFX_ROOT}/include
  ${OFX_ROOT}/Support/include
  ${OFX_ROOT}/Support/Plugins/include
)
set_target_properties(TextAnimator PROPERTIES PREFIX "" SUFFIX ".ofx")

if(MSVC)
  # Static CRT: an OFX bundle is loaded by a host that will not have our
  # redistributables, and a missing VCRUNTIME dll fails as a silent no-load.
  set_property(TARGET TextAnimator rta_core PROPERTY
               MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(TextAnimator PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ------------------------------------------------------------------- CUDA path
# Optional: without a CUDA toolkit the plugin still builds and runs on the CPU.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Must be set BEFORE enable_language: that call installs a default (75 here),
  # after which a "if(NOT DEFINED ...)" guard silently never fires and the
  # plugin ships PTX-only code that the driver has to JIT.
  # CUDA 13 dropped everything below Turing. 120 is Blackwell (RTX 50-series).
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120 CACHE STRING "CUDA archs to build" FORCE)
  enable_language(CUDA)
  target_sources(TextAnimator PRIVATE src/cuda_compositor.cu)
  target_compile_definitions(TextAnimator PRIVATE RTA_WITH_CUDA=1)
  set_target_properties(TextAnimator PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    # Static runtime for the same reason as the CRT: a missing cudart DLL would
    # show up as the plugin silently not appearing in Resolve.
    CUDA_RUNTIME_LIBRARY Static
  )
  message(STATUS "CUDA enabled (archs: ${CMAKE_CUDA_ARCHITECTURES})")

  # CPU-vs-CUDA parity harness. The CPU compositor is the reference segtest
  # exercises; in Resolve every frame goes through the CUDA one instead.
  add_executable(cudatest tools/cudatest.cpp src/cuda_compositor.cu)
  target_link_libraries(cudatest PRIVATE rta_core)
  # cudatest.cpp is host code that includes cuda_runtime.h, so it needs the
  # toolkit headers that .cu files get for free.
  target_include_directories(cudatest PRIVATE external/stb
                             ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_compile_definitions(cudatest PRIVATE RTA_WITH_CUDA=1 _CRT_SECURE_NO_WARNINGS)
  set_target_properties(cudatest PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Static)
  if(MSVC)
    set_property(TARGET cudatest PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
else()
  target_compile_definitions(TextAnimator PRIVATE RTA_WITH_CUDA=0)
  message(STATUS "CUDA toolkit not found - building CPU-only")
endif()

# Assemble the .ofx.bundle next to the binary.
set(BUNDLE ${CMAKE_BINARY_DIR}/bundle/TextAnimator.ofx.bundle/Contents/Win64)
add_custom_command(TARGET TextAnimator POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${BUNDLE}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:TextAnimator> ${BUNDLE}/TextAnimator.ofx
  COMMENT "Assembling TextAnimator.ofx.bundle"
)

# ---------------------------------------------------------------- test harness
add_executable(segtest tools/segtest.cpp)
target_link_libraries(segtest PRIVATE rta_core)
target_include_directories(segtest PRIVATE external/stb)
if(MSVC)
  set_property(TARGET segtest PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  target_compile_definitions(segtest PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
