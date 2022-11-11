cmake_minimum_required(VERSION 3.10.2)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_NAME triangulation)
project(${PROJECT_NAME})

message("compile project : ${CMAKE_PROJECT_NAME}: ${PROJECT_NAME}")
message("current cmake src dir : ${CMAKE_CURRENT_SOURCE_DIR}")

# file(GLOB SRC_FILES 
#            "src/graph.hpp"
#            "src/graph.cpp")

#find_package(OpenCV REQUIRED PATHS "/usr/local/Cellar/opencv/4.5.5_3/" NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED PATHS "/usr/local/opt/opencv/" NO_DEFAULT_PATH)
message(${OpenCV_LIBS})
message(${OpenCV_INCLUDE_DIRS})


include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${OpenCV_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} main.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        )
