cmake_minimum_required (VERSION 2.8.11)
set(FaceVerSrzn_MAJOR_VERSION 0)
set(FaceVerSrzn_MINOR_VERSION 1)
set(FaceVerSrzn_PATCH_VERSION 0)
set(FaceVerSrzn_VERSION
  ${FaceVerSrzn_MAJOR_VERSION}.${FaceVerSrzn_MINOR_VERSION}.${FaceVerSrzn_PATCH_VERSION})
project (FaceVerSrzn)

# ---[ Using cmake scripts and modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT) 
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/install" CACHE PATH "Default install path" FORCE) 
endif() 

# set up include-directories
include_directories(
  "${PROJECT_SOURCE_DIR}"   # to find foo/foo.h
  "${PROJECT_BINARY_DIR}")  # to find foo/config.h
add_library(dlib SHARED IMPORTED)   # Dynamic import

# Add dlib
set_target_properties(dlib PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../lib/libdlib.so)
include(../dlib/cmake)  

find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
##find_package(Boost REQUIRED)
#find_package(Boost COMPONENTS filesystem system REQUIRED)
#include_directories(${Boost_INCLUDE_DIRS})
#find_package(Flann REQUIRED)
#include_directories(${FLANN_INCLUDE_DIRS})
FIND_PACKAGE( OpenMP REQUIRED)
#find_package(HDF5 REQUIRED)
#include_directories(${HDF5_INCLUDE_DIRS})

configure_file(config.h.in "${CMAKE_CURRENT_BINARY_DIR}/config.h" @ONLY)

include_directories("./include")
#link_directories("../lib")
#set(FaceVerSrzn_LIBRARIES "face_ver_srzn;caffe;dlib")
set(FaceVerSrzn_LIBRARIES "caffe;dlib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#add_executable(face_ver_srzn face_ver.cpp classifier.h classifier.cpp face_align.h face_align.cpp)
add_executable(face_ver_srzn 
    include/classifier.h 
    include/verification.h 
    include/face_align.h 
    face_ver.cpp 
    src/classifier.cpp 
    src/verification.cpp 
    src/face_align.cpp)
#target_link_libraries(face_ver_srzn ${FaceVerSrzn_LIBRARIES} ${OpenCV_LIBRARIES} ${Boost_LIBRARIES} )
#target_link_libraries(face_ver_srzn ${FaceVerSrzn_LIBRARIES} ${OpenCV_LIBRARIES} )
target_link_libraries(face_ver_srzn ${OpenCV_LIBRARIES} caffe dlib)
