cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(TensorGraph
  VERSION 0.0.1
  DESCRIPTION "TensorGraph"
  LANGUAGES C CXX)

set(DEFAULT_BUILD_TYPE "Release")

#STNDARD STUFF
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_COMMON} -ltensorflow_framework")
set(TARGET_NAME "TensorGraph")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
configure_file(config.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config.json)

# add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmakeModules/")
#list(APPEND CMAKE_MODULE_PATH "/usr/share/eigen3/cmake/")

# fix missing eigen3:
# sudo ln -s /usr/include/eigen3/unsupported  /usr/include/tensorflow/unsupported

message(STATUS ${CMAKE_MODULE_PATH})
message("TensorFlow_LIBRARIES:" )
message(STATUS  ${TensorFlow_LIBRARIES})
message("TensorFlow_INCLUDE_DIRS:")
message(STATUS ${TensorFlow_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorFlow MODULE REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


set(SRC
  ../IncludeSpdlog/spdlog.cc
  ../ConfigReader/src/configreader.cc
  src/tensorgraph.cc
  src/mainloop.cc
  src/main.cc
  )

set(HEADERS
  ../IncludeSpdlog/spdlog.h
  ../ConfigReader/src/configreader.h
  src/tensorgraph.h
  src/mainloop.h
  src/structures.h
)

add_executable(${TARGET_NAME}
    ${SRC}
    ${HEADERS}
)

set(PROJECT_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIRS})
set(PROJECT_LIBRARIES ${TensorFlow_LIBRARIES} )
include_directories(${PROJECT_INCLUDE_DIRS})

qt5_use_modules(${PROJECT_NAME} Widgets )

target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS} ${PROJECT_LIBRARIES} ${QT_LIBRARIES}  Threads::Threads )

