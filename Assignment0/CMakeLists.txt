aux_source_directory (. DIR) # '.'代表当前文件夹 这个指令将遍历当前文件夹下的所有源代码，将它作为一个 list存入变量‘DIR’中

set(IDX Assignment0) #定义一个变量‘IDX’，它的值为文件夹的名字，Assignment0

add_executable (${IDX} ${DIR}) #在整个工程里添加一个名为${IDX}的可执行文件，可执行文件的源代码是${DIR}

set(CMAKE_CXX_STANDARD 17) #配置C++ Standard

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin/${IDX}) #配置可执行文件的输出路径