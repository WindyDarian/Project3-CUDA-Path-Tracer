CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Ruoyu Fan
* Tested on:
  * Windows 10 x64 / Ubuntu 16.04 x64, i7-4720HQ @ 2.60GHz, 16GB Memory, GTX 970M 3072MB (personal laptop)
  * Visual Studio 2015 + CUDA 8.0 on Windows
  * gcc/g++ 5.4 + CUDA 8.0 on Ubuntu

### Things I have done

* Basics:
  * Path tracing diffusive and perfect specular materials
  * Original __glfw3__ lib files doesn't support __Visual Studio 2015__. I updated __glfw3__ and put the source version into the `external/` folder and configured `CMakeLists.txt` so it becomes compatible with __Visual Studio 2015__ while can also build on other compilers supported. Also upgraded CMake __FindCuda__ module to solve linker errors in CUDA 8.
  * Used `thrust::partition` and `thrust::remove_if` to compact the path segment array... but only to find that __the rendering speed after stream compaction is SLOWER__. Not yet tested the data in details, but I doubt it is due to the cost of moving `PathSegments` around. I plan to build a 0/1 array according to the termination state of the path segment array and scan/compact the 0/1 array to get an index array for forwarding the threads instead
  * Sorts by material after getting intersections. (Toggleable by changing `SORT_PATH_BY_MATERIAL` in `pathtrace.cu`)
  * Caching first intersections. (Toggleable by changing `CACHE_FIRST_BONUCE` in `pathtrace.cu`)
  * Time measurement.

### TODOs

* ~~Path tracing diffusive materials~~
* ~~Fix float number precision error~~
* ~~Perfect specular materials~~
* ~~Stream compaction~~
* compact a pointer array instead of array of not-very-small `PathSegment`s to see if there is any performance increase
* ~~sort the array of `PathSegment`s by material....~~ or again sort that pointer array?
* ~~cache first intersection~~

#### Current State
![current_screenshot_or_render](/screenshots/screenshot_current.jpg)


#### That is what I started from
![begin_screenshot](/screenshots/screenshot_begin.png)
