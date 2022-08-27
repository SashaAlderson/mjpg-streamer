cmake_minimum_required(VERSION 2.8.3)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

find_package(OpenCV COMPONENTS core imgproc highgui videoio)

set(CMAKE_BUILD_TYPE Release)
  enable_language(CXX)
  include_directories(include ${OpenCV_INCLUDE_DIRS})
  include_directories(/usr/include)

  # project("yolact" CXX)

  set(TENGINE_LIBRARY_DIR /usr/lib )

  set(TENGINE_LIBRARY
      -L${TENGINE_LIBRARY_DIR} 
      -ltengine-lite)

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -std=gnu++17")
 
  add_library(input_opencv SHARED input_opencv.cpp yolact_uint8.cpp) 
  target_link_libraries(input_opencv ${TENGINE_LIBRARY} ${OpenCV_LIBS} -pthread)
  
  install(TARGETS input_opencv)

  add_subdirectory(filters/cvfilter_cpp)
  add_subdirectory(filters/cvfilter_py)