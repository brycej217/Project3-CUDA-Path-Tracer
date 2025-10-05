CUDA Path Tracer
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Bryce Joseph
* [LinkedIn](https://www.linkedin.com/in/brycejoseph/), [GitHub](https://github.com/brycej217)
* Tested on: Windows 11, Intel(R) CORE(TM) Ultra 9 275HX @ 2.70GHz 32.0GB, NVIDIA GeFORCE RTX 5080 Laptop GPU 16384MB

# CIS 5650 Project 3 - Path Tracer
This project involved creating implementations of parallel scan and stream compaction algorithms in CUDA and testing their performance.    
Namely, a work-efficient parallel scan, naive parallel scan, and cpu and thrust implementations for testing against. 
The algorithms were based on class materials and excerpts from [GPU Gems 3 Chapter 39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda), the implementations of which I will discuss now:

## Features
* Path Tracing
* Early Termination
* Material Sorting
* Stochastic Sampled Antialiasing

* Scene Loading
* Bounding Volume Hierarchies
* Diffuse, Normal, and Roughness Texture Mapping
* Environment Mapping
* Depth of Field

## Implementation
### Naive Scan

### Work-Efficient Scan
