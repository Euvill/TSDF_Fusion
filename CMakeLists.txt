CMAKE_MINIMUM_REQUIRED(VERSION 3.10.2)
 
PROJECT(KLTtracker)
 
SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_compile_options(-std=c++14)
add_definitions(-std=c++14)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

include_directories("/usr/local/include/eigen3")
include_directories(include)

find_package(OpenCV 3.4.16 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES ${OpenCV_LIBRARIES})

find_package(PCL 1.7 REQUIRED)
list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")

include_directories(${PCL_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES ${PCL_LIBRARIES})

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE ALL_SRCS "${PROJECT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE NODE_SRCS "${PROJECT_SOURCE_DIR}/src/*_app.cpp")
file(GLOB_RECURSE COMPILE1_SRCS "${PROJECT_SOURCE_DIR}/*CompilerId.cpp")
file(GLOB_RECURSE COMPILE2_SRCS "${PROJECT_SOURCE_DIR}/*CheckVersion.cpp")
file(GLOB_RECURSE COMPILE3_SRCS "${PROJECT_SOURCE_DIR}/*TryFlag.cpp")
list(REMOVE_ITEM ALL_SRCS ${NODE_SRCS} ${COMPILE1_SRCS} ${COMPILE2_SRCS} ${COMPILE3_SRCS})

add_executable(fusion_app src/fusion_app.cpp ${ALL_SRCS})
target_link_libraries(fusion_app ${ALL_TARGET_LIBRARIES} OpenMP::OpenMP_CXX)

