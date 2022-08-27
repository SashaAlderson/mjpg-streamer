
MJPG_STREAMER_PLUGIN_OPTION(cvfilter_cpp "OpenCV example filter")
MJPG_STREAMER_PLUGIN_COMPILE(cvfilter_cpp filter_cpp.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cvfilter_cpp ${OpenCV_LIBS})
