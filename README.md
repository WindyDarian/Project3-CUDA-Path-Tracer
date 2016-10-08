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
  * Used `thrust::remove_if` to compact the path segment array... but only to find that __the rendering speed after stream compaction is SLOWER__. Not yet tested the data in details, but I doubt it is due to the cost of moving `PathSegments` around. I plan to build a 0/1 array according to the termination state of the path segment array and scan/compact the 0/1 array to get an index array for forwarding the threads instead
  * Sorts by material after getting intersections. (Toggleable by changing `SORT_PATH_BY_MATERIAL` in `pathtrace.cu`)
  * Caching first intersections. (Toggleable by changing `CACHE_FIRST_INTERSECTION` in `pathtrace.cu`)
  * Time measurement.

### TODOs

* ~~Path tracing diffusive materials~~
* ~~Fix float number precision error~~
* ~~Perfect specular materials~~
* ~~Stream compaction~~
* compact a pointer array instead of array of not-very-small `PathSegment`s to see if there is any performance increase
* ~~sort the array of `PathSegment`s by material....~~ or again sort that pointer array?
* ~~cache first intersection~~
* Test copy values instead of references for some data (for example intersection of current path segment) at intersection/shading stage

### Performance Tests
#### Core Features

Since I may abandon some of the features during development (such as depth of field or somewhat sampling), I tested the results by toggling on and off `ENABLE_STREAM_COMPACTION`, `SORT_PATH_BY_MATERIAL` and `CACHE_FIRST_INTERSECTION`, based on this commit: [`core_features`](https://github.com/WindyDarian/Project3-CUDA-Path-Tracer/releases/tag/core_features)

* `ENABLE_STREAM_COMPACTION`: whether doing stream compaction using `thrust::remove_if` after shading.  (Which, however, involves moving somewhat big objects around and slows the program)
* `SORT_PATH_BY_MATERIAL`: whether sorting paths by material. (by thrust::sort_by_key). Currently sorting both the `ShadeableIntersections` and `PathSegments` arrays. (Which involves moving somewhat big objects around and slows the program)
* `CACHE_FIRST_INTERSECTION` : the intersections from camera rays are cached. It sightly improves performance.

The results are as follows (tested with default "cornell.txt" scene, with diffusive walls and a perfect specular sphere):

| Test Case Id | ENABLE_STREAM_COMPACTION | SORT_PATH_BY_MATERIAL | CACHE_FIRST_INTERSECTION | Time for 5000 iterations (s) | Iterations per second |
|--------------|--------------------------|-----------------------|--------------------------|------------------------------|-----------------------|
| 000          | OFF                      | OFF                   | OFF                      | 188.283                      | 26.5557               |
| 100          | ON                       | OFF                   | OFF                      | 332.301                      | 15.0466               |
| 010          | OFF                      | ON                    | OFF                      | 1356.74                      | 3.68532               |
| 001          | OFF                      | OFF                   | ON                       | 169.077                      | 29.5723               |
| 110          | ON                       | ON                    | OFF                      | 970.748                      | 5.15067               |
| 111          | ON                       | ON                    | ON                       | 956.356                      | 5.22818               |

![chart_core_features](/test_results/chart_core_features.png)

Interestingly, while both `ENABLE_STREAM_COMPACTION` (100) and `SORT_PATH_BY_MATERIAL` (010) are slower than the naive way (000), enabling them both (110) is faster than enabling `SORT_PATH_BY_MATERIAL` (010) only. That is because enabling stream compaction reduces a lot work of sorting.

#### Additional Core Feature Tests

Doing it.

#### Current State
![current_screenshot_or_render](/rendered_images/cornell.2016-10-03_04-33-43z.5000samp.png)


#### That is what I started from
![begin_screenshot](/screenshots/screenshot_begin.png)
