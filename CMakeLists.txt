cmake_minimum_required (VERSION 2.8)
project (SNN)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

include_directories(${PROJECT_SOURCE_DIR}/src/SNN/utils)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Interfaces)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Neurons)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Synapses)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Gradients)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Optimizers)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Networks)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Options)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Visualizations)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Kernels/CPU)
include_directories(${PROJECT_SOURCE_DIR}/src/SNN/Kernels/GPU)
include_directories(${PROJECT_SOURCE_DIR}/src/mnist-loader/include/)

set(FloatType double)
set(FloatTypeSize 64)
#set(FloatType float)
#set(FloatTypeSize 32)

set(CUDA ON)
if (CUDA)
    set(DEFINITIONS -DFloatType=${FloatType} -DFloatTypeSize=${FloatTypeSize})
else()
    set(DEFINITIONS -DFloatType=${FloatType} -DFloatTypeSize=${FloatTypeSize} -DNO_CUDA)
endif()

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

if (CUDA)
    find_package(CUDA QUIET REQUIRED)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("DEBUG BUILD")
    if (CUDA)
        set(CUDA_NVCC_FLAGS "-g -G -std=c++11 -arch=sm_61 --compiler-options -Wall --compiler-options -Wextra -maxrregcount 64")
    endif()
    set(OPTIMIZER_FLAGS -g -std=c++11 -Wall -Wextra)
else()
    if (CUDA)
        set(CUDA_NVCC_FLAGS "-std=c++11 -O2 -arch=sm_75 --compiler-options -Wall --compiler-options -Wextra -maxrregcount 64 -res-usage")
    endif()
    set(OPTIMIZER_FLAGS -O2 -g -std=c++11 -Wall -Wextra)
endif()

add_executable(
    RobotArmPredictor
    ${PROJECT_SOURCE_DIR}/RobotArmPredictor
    ${PROJECT_SOURCE_DIR}/src/SNN/Networks/RealisticRobotArmV2PredictorNetwork
    ${PROJECT_SOURCE_DIR}/src/SNN/Networks/RealisticRobotArmV3PredictorNetwork
    ${PROJECT_SOURCE_DIR}/src/SNN/Networks/LongShortTermMemoryEligibilityNetwork
    ${PROJECT_SOURCE_DIR}/src/SNN/utils/Image
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTK3DWindow
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTK3DObject
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/RealisticRobotArmSimulationV2
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/RealisticRobotArmSimulationV3
    ${PROJECT_SOURCE_DIR}/src/SNN/utils/utils
    ${PROJECT_SOURCE_DIR}/src/SNN/utils/Threads
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTKWindow
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTKPlotWindow
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTKPlotObject
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTKInteractor
)
add_executable(
    RobotArmInference
    ${PROJECT_SOURCE_DIR}/RobotArmInference
    ${PROJECT_SOURCE_DIR}/src/SNN/Networks/RealisticRobotArmV2InferenceNetwork
    ${PROJECT_SOURCE_DIR}/src/SNN/Networks/RealisticRobotArmV3InferenceNetwork
    ${PROJECT_SOURCE_DIR}/src/SNN/Networks/LongShortTermMemoryEligibilityNetwork
    ${PROJECT_SOURCE_DIR}/src/SNN/utils/Image
    ${PROJECT_SOURCE_DIR}/src/SNN/utils/arduino-serial-lib
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/RealisticRobotArmSimulationV2
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/RealisticRobotArmSimulationV3
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTK3DWindow
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTK3DObject
    ${PROJECT_SOURCE_DIR}/src/SNN/utils/utils
    ${PROJECT_SOURCE_DIR}/src/SNN/utils/Threads
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTKWindow
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTKPlotWindow
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTKPlotObject
    ${PROJECT_SOURCE_DIR}/src/SNN/Visualizations/VTKInteractor
)

if (CUDA)

    cuda_add_library(
        gpu_kernels STATIC
        ${PROJECT_SOURCE_DIR}/src/SNN/Kernels/GPU/FullyConnectedFixedBroadcastKernelCaller.cu
        ${PROJECT_SOURCE_DIR}/src/SNN/Kernels/GPU/LongShortTermMemoryKernelCaller.cu
        ${PROJECT_SOURCE_DIR}/src/SNN/Kernels/GPU/LongShortTermMemorySparseKernelCaller.cu
    )
endif()

target_compile_definitions(RobotArmPredictor PUBLIC ${DEFINITIONS})
target_compile_definitions(RobotArmInference PUBLIC ${DEFINITIONS})
if (CUDA)
    target_compile_definitions(gpu_kernels PUBLIC ${DEFINITIONS})
endif()

target_compile_options(RobotArmPredictor PUBLIC ${OPTIMIZER_FLAGS})
target_compile_options(RobotArmInference PUBLIC ${OPTIMIZER_FLAGS})
if (CUDA)
    target_compile_options(gpu_kernels PUBLIC ${OPTIMIZER_FLAGS})
endif()

if(VTK_LIBRARIES)
  if (CUDA)
    target_link_libraries(RobotArmPredictor ${VTK_LIBRARIES} pthread gomp gpu_kernels IL)
    target_link_libraries(RobotArmInference ${VTK_LIBRARIES} pthread gomp gpu_kernels IL)
  else()
    target_link_libraries(RobotArmPredictor ${VTK_LIBRARIES} pthread gomp IL)
    target_link_libraries(RobotArmInference ${VTK_LIBRARIES} pthread gomp IL)
  endif()
else()
  target_link_libraries(RobotArmPredictor vtkHybrid vtkWidgets pthread gpu_kernels IL)
  target_link_libraries(RobotArmInference vtkHybrid vtkWidgets pthread gpu_kernels IL)
endif()
