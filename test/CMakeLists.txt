aux_source_directory (. DIR) # '.'代表当前文件夹 这个指令将遍历当前文件夹下的所有源代码，将它作为一个 list存入变量‘DIR’中

set(IDX test) #定义一个变量‘IDX’，它的值为文件夹的名字，test

add_executable (${IDX} ${DIR}) #在整个工程里添加一个名为${IDX}的可执行文件，可执行文件的源代码是${DIR}

find_package(OpenCV REQUIRED)
target_link_libraries(${IDX} ${OpenCV_LIBS}) #在第二步里提到的配置OPENCV的方法

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin/${IDX}) #配置可执行文件的输出路径
set(CMAKE_CXX_STANDARD 17) #配置C++ Standard
add_definitions(-std=c++17)
