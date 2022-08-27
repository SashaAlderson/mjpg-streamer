#!/bin/bash
./mjpg_streamer -i "libinput_opencv.so -d 0 -filter /home/khadas/workspace/mjpg-streamer/mjpg-streamer-experimental/cvfilter_cpp.so" -o "output_http.so" 
# ./mjpg_streamer -i "libinput_opencv.so -d test.mp4 -filter /home/khadas/workspace/mjpg-streamer/mjpg-streamer-experimental/cvfilter_cpp.so" -o "output_http.so" 