CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ruoyu Fan
* Tested on:
  * Windows 10 x64 / Ubuntu 16.04 x64, i7-4720HQ @ 2.60GHz, 16GB Memory, GTX 970M 3072MB (personal laptop)
  * Visual Studio 2015 + CUDA 8.0 on Windows
  * gcc/g++ 5.4 + CUDA 8.0 on Ubuntu

### Things I have done

* Path tracing diffusive materials
* Original __glfw3__ lib files doesn't support __Visual Studio 2015__. I updated __glfw3__ and put the source version into the `external/` folder and configured `CMakeLists.txt` so it becomes compatible with __Visual Studio 2015__ while can also build on other compilers supported.  

### TODOs

* ~~Path tracing diffusive materials~~

* Fix float error.

#### Current State
![current_screenshot](/screenshots/screenshot_current.png)


#### That is what I started from
![begin_screenshot](/screenshots/screenshot_begin.png)


*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.
