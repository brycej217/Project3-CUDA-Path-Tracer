#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct AABB
{
    glm::vec3 min;
    glm::vec3 max;


    __host__ __device__ void swap(float& a, float& b)
    {
        float temp = a;
        a = b;
        b = temp;
    }

    __host__ __device__ bool hit(Ray r)
    {
        float tmin = -1e38f;
        float tmax = 1e38f;

        for (int xyz = 0; xyz < 3; ++xyz)
        {
            float invD = 1.0f / r.direction[xyz];
            float t0 = (min[xyz] - r.origin[xyz]) * invD;
            float t1 = (max[xyz] - r.origin[xyz]) * invD;

            if (invD < 0.0f) swap(t0, t1);

            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }

        return true;
    }
};

struct Geom
{
    AABB aabb;
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    glm::vec3 center;

    glm::vec3 getCenter()
    {
        return glm::vec3(transform * glm::vec4(0.0f));
    }
};

struct BVHNode
{
    AABB aabb;
    int left;
    int right;
    int startGeom; // index of first geom in geom array
    int numGeoms;
};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    bool hasSpecular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 radiance;
    glm::vec3 throughput;
    int pixelIndex;
    int remainingBounces;
    float t;
};

struct Sample
{
    glm::vec3 lo;
    Ray wi;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
