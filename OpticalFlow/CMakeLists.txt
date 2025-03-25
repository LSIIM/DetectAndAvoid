cmake_minimum_required (VERSION 3.6)
project(detectandavoidgd45)

find_package(OpenCV REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

set(CMAKE_CXX_FLAGS "-g /O2 -ggdb -Wall -fpermissive")
#set(CMAKE_CXX_FLAGS "-O3")

add_executable(detectandavoid src/main_opticalLucas.cpp ../FuzzyCMeans/fuzzy_clustering.cpp)

#target_link_libraries(detectandavoid ShadowCounter)
#target_link_libraries(detectandavoid Background)
target_link_libraries(detectandavoid ${OpenCV_LIBS})


#add_library(detectandavoid src/shadowCounter.cpp)
#add_library(detectandavoid src/background.cpp)
