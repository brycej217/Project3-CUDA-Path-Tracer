#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

#define STREAM_COMPACT
#define MAT_SORT
#define ERRORCHECK 1

using namespace glm;
using namespace std;

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;

static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static cudaTextureObject_t* dev_tex = NULL;
static cudaTextureObject_t* dev_env = NULL;
static BVHNode* dev_nodes = NULL;
static Triangle* dev_tris = NULL;
static int* dev_keys1 = NULL;
static int* dev_keys2 = NULL;
static thrust::device_ptr<PathSegment> thrust_paths;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need

    // MAT SORT

    cudaMalloc(&dev_keys1, pixelcount * sizeof(int));
    cudaMalloc(&dev_keys2, pixelcount * sizeof(int));
    thrust_paths = thrust::device_pointer_cast(dev_paths);

    // BVH Nodes

    cudaMalloc(&dev_nodes, scene->nodes.size() * sizeof(BVHNode));
    cudaMemcpy(dev_nodes, scene->nodes.data(), scene->nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);

    // Triangles
    cudaMalloc(&dev_tris, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_tris, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    // Textures
    cudaMalloc(&dev_tex, scene->textures.size() * sizeof(cudaTextureObject_t));
    cudaMemcpy(dev_tex, scene->textures.data(), scene->textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

    // Env
    cudaMalloc(&dev_env, sizeof(cudaTextureObject_t));
    cudaMemcpy(dev_env, &scene->env, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_nodes);
    cudaFree(dev_keys1);
    cudaFree(dev_keys2);

    cudaFree(dev_tris);
    cudaFree(dev_tex);
    cudaFree(dev_env);
    checkCUDAError("pathtraceFree");
}

__device__ glm::vec2 sampleDisk(glm::vec2 uv)
{
    float u = uv.x;
    float v = uv.y;

    glm::vec2 uOffset = 2 * u - glm::vec2(1.0f, 1.0f);
    if (uOffset.x == 0 && uOffset.y == 0) 
    {
        return glm::vec2(0.0f, 0.0f);
    }

    float theta, r;
    if (abs(uOffset.x) > abs(uOffset.y))
    {
        r = uOffset.x;
        theta = (PI / 4) * (uOffset.y / uOffset.x);
    }
    else
    {
        r = uOffset.y;
        theta = (PI / 2) - (PI / 4) * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(cos(theta), sin(theta));
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, bool dof)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;


    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
        segment.ray.origin = cam.position;

        // set default payload values
        segment.radiance = vec3(0.0f);
        segment.throughput = vec3(1.0f);
        segment.compact = false;
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;

        // get noise values
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> u(-0.5f, 0.5f);
        thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
        float xNoise = u(rng);
        float yNoise = u(rng);

        // TODO: implement antialiasing by jittering the ray
        glm::vec3 dir = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)(x + xNoise) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)(y + yNoise) - (float)cam.resolution.y * 0.5f)
        );

        // depth of field
        if (dof)
        {
            dir *= cam.focalDistance; // scale dir to focus distance
            glm::vec2 aperture = sampleDisk(glm::vec2(u01(rng), u01(rng))) * cam.lensRadius; // scale sampled disk point to lens radius
            glm::vec3 apViewSpace = aperture.x * cam.right + aperture.y * cam.up; // get view space coordinates of sampled point

            segment.ray.origin = cam.position + apViewSpace;
            segment.ray.direction = glm::normalize(dir - apViewSpace);
        }
        else
        {
            segment.ray.direction = dir;
        }
    }
}


__host__ __device__ float intersectGeoms(Triangle* geoms, int num_geoms, PathSegment* paths, int pathId, int start, int end, float& t_min, int& hit_geom_index, glm::vec3& intersect_point, glm::vec3& normal, glm::vec2& uv)
{
    float t;
    bool outside = true;

    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_uv;

    for (int i = start; i < start + end; i++)
    {
        Triangle& geom = geoms[i];

        t = triangleIntersectionTest(geom, paths[pathId].ray, tmp_intersect, tmp_normal, outside, tmp_uv);

        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            hit_geom_index = i;
            intersect_point = tmp_intersect;
            normal = tmp_normal;
            uv = tmp_uv;
        }
    }
    return t_min;
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    BVHNode* nodes,
    int num_nodes,
    Triangle* geoms,
    int geoms_size,
    ShadeableIntersection* intersections,
    int* keys1,
    int* keys2,
    bool bvh)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uv;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        // traverse BVH
        if (bvh)
        {
            int stack[64];
            int sp = 0;
            stack[sp++] = 0;

            while (sp)
            {
                int i = stack[--sp];

                if (i < 0 || !nodes[i].aabb.hit(pathSegments[path_index].ray)) continue;

                if (nodes[i].numGeoms > 0)
                {
                    intersectGeoms(geoms, geoms_size, pathSegments, path_index, nodes[i].startGeom, nodes[i].numGeoms, t_min, hit_geom_index, intersect_point, normal, uv);
                    continue;
                }

                stack[sp++] = nodes[i].right;
                stack[sp++] = nodes[i].left;
            }
        }
        else
        {
            intersectGeoms(geoms, geoms_size, pathSegments, path_index, 0, geoms_size, t_min, hit_geom_index, intersect_point, normal, uv);
        }


        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialId;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].uv = uv;

            keys1[path_index] = geoms[hit_geom_index].materialId;
            keys2[path_index] = geoms[hit_geom_index].materialId;
        }
    }
}

__global__ void shadeIntersection(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    cudaTextureObject_t* textures,
    cudaTextureObject_t* env,
    bool hasEnv,
    bool enableEnv,
    float envGain,
    bool texturing)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if we have an intersection
        {
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegments[idx].remainingBounces);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];

            if (material.emittance > 0.0f)
            {
                pathSegments[idx].radiance = material.color * material.emittance;
            }
            else
            {
                pathSegments[idx].radiance = vec3(0.0f);
                
                Sample sample = sampleBSDF(pathSegments[idx], intersection, material, rng, textures, texturing);

                pathSegments[idx].throughput *= sample.lo;

                // set new ray
                pathSegments[idx].ray = sample.wi;
                pathSegments[idx].remainingBounces--;
            }
        }
        else // if not set throughput to 0
        {
            if (hasEnv && enableEnv)
            {
                pathSegments[idx].radiance = pathSegments[idx].throughput * sampleEnv(pathSegments[idx], env) * envGain;
            }
            else
            {
                pathSegments[idx].throughput = vec3(0.0f);
                pathSegments[idx].compact = true;
            }
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.throughput * iterationPath.radiance;
    }
}

struct earlyTerm
{
    __host__ __device__ bool operator()(PathSegment path) const
    {
        return path.remainingBounces < 0 || path.compact;
    }
};

void streamCompact(int& num_paths)
{
    auto begin = thrust::device_pointer_cast(dev_paths);
    auto end = begin + num_paths;

    auto new_end = thrust::remove_if(begin, end, earlyTerm{});
    num_paths = static_cast<int>(new_end - begin);
}

void materialSort(int num_paths)
{
    auto thrust_paths = thrust::device_pointer_cast(dev_paths);
    auto thrust_intersections = thrust::device_pointer_cast(dev_intersections);
    auto thrust_keys1 = thrust::device_pointer_cast(dev_keys1);
    auto thrust_keys2 = thrust::device_pointer_cast(dev_keys2);

    thrust::sort_by_key(thrust_keys1, thrust_keys1 + num_paths, thrust_paths);
    thrust::sort_by_key(thrust_keys2, thrust_keys2 + num_paths, thrust_intersections);
}

void pathtrace(uchar4* pbo, int iter)
{
    // trace setup
    bool earlyTermination = hst_scene->streamCompaction;
    bool matSort = hst_scene->matSort;
    bool enableEnv = hst_scene->environmentMapping;

    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    const int blockSize1d = 128;

    // generate rays
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, hst_scene->dof);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;
    while (num_paths > 0 && depth < traceDepth)
    {
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // determine intersections
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_nodes,
            hst_scene->num_nodes,
            dev_tris,
            hst_scene->triangles.size(),
            dev_intersections,
            dev_keys1,
            dev_keys2,
            hst_scene->bvh
            );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        if (matSort)
        {
            materialSort(num_paths);
        }

        // shade intersections
        shadeIntersection << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_tex,
            dev_env,
            hst_scene->hasEnv,
            enableEnv,
            hst_scene->envGain,
            hst_scene->texturing
            );

        // stream compaction
        if (earlyTermination)
        {
            streamCompact(num_paths);
        }

        depth++;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}