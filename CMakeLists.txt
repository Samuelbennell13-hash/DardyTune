cmake_minimum_required(VERSION 3.15)
project(DardyTune VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE_DIR is passed by GitHub Actions as -DJUCE_DIR=...
# For local builds you can set it yourself.
if (NOT DEFINED JUCE_DIR)
  message(FATAL_ERROR "JUCE_DIR not set. Pass -DJUCE_DIR=path/to/JUCE")
endif()

add_subdirectory(${JUCE_DIR} JUCE)

juce_add_plugin(DardyTune
    COMPANY_NAME "Sam_B"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE SamB
    PLUGIN_CODE Dtdy
    FORMATS VST3
    PRODUCT_NAME "DardyTune"
)

target_sources(DardyTune PRIVATE
    DardyTune.cpp
)

target_compile_definitions(DardyTune PRIVATE
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(DardyTune PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
)
