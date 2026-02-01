cmake_minimum_required(VERSION 3.15)

project(DardyTune VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)

add_library(DardyTune SHARED
    DardyTune.cpp
)

# JUCE will be linked later on PC
