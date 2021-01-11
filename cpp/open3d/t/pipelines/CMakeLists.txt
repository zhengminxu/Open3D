# Build
set(PIPELINES_SRC
    registration/Registration.cpp
    registration/TransformationEstimation.cpp
    kernel/TransformationConverter.cpp
)

set(PIPELINES_CUDA_SRC
    kernel/TransformationConverter.cu
)

if(BUILD_CUDA_MODULE)
    set(PIPELINES_SRC
        ${PIPELINES_SRC}
        ${PIPELINES_CUDA_SRC}
    )
endif()

# Create object library
add_library(tpipelines OBJECT ${PIPELINES_SRC})
open3d_show_and_abort_on_warning(tpipelines)
open3d_set_global_properties(tpipelines)
open3d_set_open3d_lib_properties(tpipelines)
open3d_link_3rdparty_libraries(tpipelines)

if(BUILD_CUDA_MODULE)
    target_include_directories(tpipelines SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
