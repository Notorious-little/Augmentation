cmake_minimum_required(VERSION 2.8)

project(Augmentation)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_EXE img_create.cpp methods.cpp OO_methods.cpp gradient_contour.cpp) #img_create==main

set(OpenCV_DIR /home/ruslan/Загрузки/opencv/build)

set(SOURCE_LIB methods.cpp OO_methods.cpp gradient_contour.cpp)

add_library(methods STATIC ${SOURCE_LIB})	

add_executable(run ${SOURCE_EXE})

target_link_libraries(run ${OpenCV_LIBS} methods)
