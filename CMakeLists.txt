cmake_minimum_required(VERSION 3.10)
project(fast-neural-style)

if(MSVC)
    add_compile_options(/utf-8)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} cpp/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
