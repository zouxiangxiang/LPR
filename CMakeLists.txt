#1.cmake verson
cmake_minimum_required(VERSION 2.8)

set (DIR ./ncnn/install)

#2.project name
project(demo C CXX)

#opencv
find_package(OpenCV REQUIRED)

#3.set environment variable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#4. include
include_directories(include ${DIR}/include/ncnn
)

# ncnn lib
add_library( libncnn STATIC IMPORTED )
set_target_properties(
        libncnn
        PROPERTIES IMPORTED_LOCATION
        ${DIR}/lib/libncnn.a
)

# add OPENMP support 
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(SRC_DETECTION src/PlateDetection.cpp  src/util.h include/PlateDetection.h)
set(SRC_FINEMAPPING src/FineMapping.cpp )
set(SRC_FASTDESKEW src/FastDeskew.cpp )
set(SRC_SEGMENTATION src/PlateSegmentation.cpp )
set(SRC_RECOGNIZE src/Recognizer.cpp src/CNNRecognizer.cpp)

set(SRC_PIPLINE src/Pipeline.cpp)

set(SRC_SEGMENTATIONFREE src/SegmentationFreeRecognizer.cpp )
set(SRC_mobilelpr src/mobilelpr.cpp )
set(SRC_UltraFace src/UltraFace.cpp include/UltraFace.hpp)
#6.source directory

add_executable(main ${SRC_DETECTION} ${SRC_FINEMAPPING} ${SRC_FASTDESKEW} ${SRC_SEGMENTATION} ${SRC_RECOGNIZE} ${SRC_PIPLINE} ${SRC_SEGMENTATIONFREE} ${SRC_mobilelpr} ${SRC_UltraFace} src/LPR.cpp)

target_link_libraries(main 
                      libncnn
                      ${OpenCV_LIBS}
					  )
