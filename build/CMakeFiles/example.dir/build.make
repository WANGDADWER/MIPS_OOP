# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wang/Desktop/MIPS_OOP-main

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wang/Desktop/MIPS_OOP-main/build

# Include any dependencies generated for this target.
include CMakeFiles/example.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/example.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/example.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/example.dir/flags.make

CMakeFiles/example.dir/src/CEOs.cpp.o: CMakeFiles/example.dir/flags.make
CMakeFiles/example.dir/src/CEOs.cpp.o: ../src/CEOs.cpp
CMakeFiles/example.dir/src/CEOs.cpp.o: CMakeFiles/example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/Desktop/MIPS_OOP-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/example.dir/src/CEOs.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/example.dir/src/CEOs.cpp.o -MF CMakeFiles/example.dir/src/CEOs.cpp.o.d -o CMakeFiles/example.dir/src/CEOs.cpp.o -c /home/wang/Desktop/MIPS_OOP-main/src/CEOs.cpp

CMakeFiles/example.dir/src/CEOs.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example.dir/src/CEOs.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/Desktop/MIPS_OOP-main/src/CEOs.cpp > CMakeFiles/example.dir/src/CEOs.cpp.i

CMakeFiles/example.dir/src/CEOs.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example.dir/src/CEOs.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/Desktop/MIPS_OOP-main/src/CEOs.cpp -o CMakeFiles/example.dir/src/CEOs.cpp.s

CMakeFiles/example.dir/src/MIPS.cpp.o: CMakeFiles/example.dir/flags.make
CMakeFiles/example.dir/src/MIPS.cpp.o: ../src/MIPS.cpp
CMakeFiles/example.dir/src/MIPS.cpp.o: CMakeFiles/example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/Desktop/MIPS_OOP-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/example.dir/src/MIPS.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/example.dir/src/MIPS.cpp.o -MF CMakeFiles/example.dir/src/MIPS.cpp.o.d -o CMakeFiles/example.dir/src/MIPS.cpp.o -c /home/wang/Desktop/MIPS_OOP-main/src/MIPS.cpp

CMakeFiles/example.dir/src/MIPS.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example.dir/src/MIPS.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/Desktop/MIPS_OOP-main/src/MIPS.cpp > CMakeFiles/example.dir/src/MIPS.cpp.i

CMakeFiles/example.dir/src/MIPS.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example.dir/src/MIPS.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/Desktop/MIPS_OOP-main/src/MIPS.cpp -o CMakeFiles/example.dir/src/MIPS.cpp.s

CMakeFiles/example.dir/src/Utilities.cpp.o: CMakeFiles/example.dir/flags.make
CMakeFiles/example.dir/src/Utilities.cpp.o: ../src/Utilities.cpp
CMakeFiles/example.dir/src/Utilities.cpp.o: CMakeFiles/example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/Desktop/MIPS_OOP-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/example.dir/src/Utilities.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/example.dir/src/Utilities.cpp.o -MF CMakeFiles/example.dir/src/Utilities.cpp.o.d -o CMakeFiles/example.dir/src/Utilities.cpp.o -c /home/wang/Desktop/MIPS_OOP-main/src/Utilities.cpp

CMakeFiles/example.dir/src/Utilities.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example.dir/src/Utilities.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/Desktop/MIPS_OOP-main/src/Utilities.cpp > CMakeFiles/example.dir/src/Utilities.cpp.i

CMakeFiles/example.dir/src/Utilities.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example.dir/src/Utilities.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/Desktop/MIPS_OOP-main/src/Utilities.cpp -o CMakeFiles/example.dir/src/Utilities.cpp.s

CMakeFiles/example.dir/src/fast_copy.c.o: CMakeFiles/example.dir/flags.make
CMakeFiles/example.dir/src/fast_copy.c.o: ../src/fast_copy.c
CMakeFiles/example.dir/src/fast_copy.c.o: CMakeFiles/example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/Desktop/MIPS_OOP-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/example.dir/src/fast_copy.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/example.dir/src/fast_copy.c.o -MF CMakeFiles/example.dir/src/fast_copy.c.o.d -o CMakeFiles/example.dir/src/fast_copy.c.o -c /home/wang/Desktop/MIPS_OOP-main/src/fast_copy.c

CMakeFiles/example.dir/src/fast_copy.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/example.dir/src/fast_copy.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/wang/Desktop/MIPS_OOP-main/src/fast_copy.c > CMakeFiles/example.dir/src/fast_copy.c.i

CMakeFiles/example.dir/src/fast_copy.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/example.dir/src/fast_copy.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/wang/Desktop/MIPS_OOP-main/src/fast_copy.c -o CMakeFiles/example.dir/src/fast_copy.c.s

CMakeFiles/example.dir/src/fht_sse.c.o: CMakeFiles/example.dir/flags.make
CMakeFiles/example.dir/src/fht_sse.c.o: ../src/fht_sse.c
CMakeFiles/example.dir/src/fht_sse.c.o: CMakeFiles/example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/Desktop/MIPS_OOP-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/example.dir/src/fht_sse.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/example.dir/src/fht_sse.c.o -MF CMakeFiles/example.dir/src/fht_sse.c.o.d -o CMakeFiles/example.dir/src/fht_sse.c.o -c /home/wang/Desktop/MIPS_OOP-main/src/fht_sse.c

CMakeFiles/example.dir/src/fht_sse.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/example.dir/src/fht_sse.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/wang/Desktop/MIPS_OOP-main/src/fht_sse.c > CMakeFiles/example.dir/src/fht_sse.c.i

CMakeFiles/example.dir/src/fht_sse.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/example.dir/src/fht_sse.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/wang/Desktop/MIPS_OOP-main/src/fht_sse.c -o CMakeFiles/example.dir/src/fht_sse.c.s

CMakeFiles/example.dir/src/header.cpp.o: CMakeFiles/example.dir/flags.make
CMakeFiles/example.dir/src/header.cpp.o: ../src/header.cpp
CMakeFiles/example.dir/src/header.cpp.o: CMakeFiles/example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/Desktop/MIPS_OOP-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/example.dir/src/header.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/example.dir/src/header.cpp.o -MF CMakeFiles/example.dir/src/header.cpp.o.d -o CMakeFiles/example.dir/src/header.cpp.o -c /home/wang/Desktop/MIPS_OOP-main/src/header.cpp

CMakeFiles/example.dir/src/header.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example.dir/src/header.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/Desktop/MIPS_OOP-main/src/header.cpp > CMakeFiles/example.dir/src/header.cpp.i

CMakeFiles/example.dir/src/header.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example.dir/src/header.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/Desktop/MIPS_OOP-main/src/header.cpp -o CMakeFiles/example.dir/src/header.cpp.s

CMakeFiles/example.dir/src/main.cpp.o: CMakeFiles/example.dir/flags.make
CMakeFiles/example.dir/src/main.cpp.o: ../src/main.cpp
CMakeFiles/example.dir/src/main.cpp.o: CMakeFiles/example.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wang/Desktop/MIPS_OOP-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/example.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/example.dir/src/main.cpp.o -MF CMakeFiles/example.dir/src/main.cpp.o.d -o CMakeFiles/example.dir/src/main.cpp.o -c /home/wang/Desktop/MIPS_OOP-main/src/main.cpp

CMakeFiles/example.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wang/Desktop/MIPS_OOP-main/src/main.cpp > CMakeFiles/example.dir/src/main.cpp.i

CMakeFiles/example.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wang/Desktop/MIPS_OOP-main/src/main.cpp -o CMakeFiles/example.dir/src/main.cpp.s

# Object files for target example
example_OBJECTS = \
"CMakeFiles/example.dir/src/CEOs.cpp.o" \
"CMakeFiles/example.dir/src/MIPS.cpp.o" \
"CMakeFiles/example.dir/src/Utilities.cpp.o" \
"CMakeFiles/example.dir/src/fast_copy.c.o" \
"CMakeFiles/example.dir/src/fht_sse.c.o" \
"CMakeFiles/example.dir/src/header.cpp.o" \
"CMakeFiles/example.dir/src/main.cpp.o"

# External object files for target example
example_EXTERNAL_OBJECTS =

example.cpython-310-x86_64-linux-gnu.so: CMakeFiles/example.dir/src/CEOs.cpp.o
example.cpython-310-x86_64-linux-gnu.so: CMakeFiles/example.dir/src/MIPS.cpp.o
example.cpython-310-x86_64-linux-gnu.so: CMakeFiles/example.dir/src/Utilities.cpp.o
example.cpython-310-x86_64-linux-gnu.so: CMakeFiles/example.dir/src/fast_copy.c.o
example.cpython-310-x86_64-linux-gnu.so: CMakeFiles/example.dir/src/fht_sse.c.o
example.cpython-310-x86_64-linux-gnu.so: CMakeFiles/example.dir/src/header.cpp.o
example.cpython-310-x86_64-linux-gnu.so: CMakeFiles/example.dir/src/main.cpp.o
example.cpython-310-x86_64-linux-gnu.so: CMakeFiles/example.dir/build.make
example.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libboost_container.so.1.74.0
example.cpython-310-x86_64-linux-gnu.so: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
example.cpython-310-x86_64-linux-gnu.so: /usr/lib/x86_64-linux-gnu/libpthread.a
example.cpython-310-x86_64-linux-gnu.so: CMakeFiles/example.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wang/Desktop/MIPS_OOP-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX shared module example.cpython-310-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/example.dir/build: example.cpython-310-x86_64-linux-gnu.so
.PHONY : CMakeFiles/example.dir/build

CMakeFiles/example.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/example.dir/cmake_clean.cmake
.PHONY : CMakeFiles/example.dir/clean

CMakeFiles/example.dir/depend:
	cd /home/wang/Desktop/MIPS_OOP-main/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wang/Desktop/MIPS_OOP-main /home/wang/Desktop/MIPS_OOP-main /home/wang/Desktop/MIPS_OOP-main/build /home/wang/Desktop/MIPS_OOP-main/build /home/wang/Desktop/MIPS_OOP-main/build/CMakeFiles/example.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/example.dir/depend
