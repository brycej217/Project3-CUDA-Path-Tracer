#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

using namespace glm;

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ vec3 reflect(vec3 V, vec3 N)
{
    return V - 2 * dot(V, N) * N;
}

__device__ vec3 sampleEnv(PathSegment& pathSegment, cudaTextureObject_t* env)
{
    vec3 dir = pathSegment.ray.direction;

    float u = atan2f(dir.z, dir.x) * (0.5f / PI) + 0.5f;
    float v = 0.5f - asinf(glm::clamp(dir.y, -1.f, 1.f)) / PI;

    float4 c = tex2D<float4>(*env, u, v);
    vec3 color = vec3(c.x, c.y, c.z);
    return glm::clamp(color, vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 1.0f, 1.0f) * 100.0f);
}

__device__ Sample sampleBSDF(
    PathSegment& pathSegment,
    ShadeableIntersection& intersection,
    const Material &m,
    thrust::default_random_engine &rng,
    cudaTextureObject_t* textures,
    bool texturing)
{
    Sample sample;
    vec2 uv = intersection.uv;

    vec3 normal = intersection.surfaceNormal;
    vec3 color = m.color;
    float roughness = m.roughness;
    float metallic = m.metallic;
    
    if (texturing)
    {
        if (m.normTexId >= 0)
        {
            float4 n = tex2D<float4>(textures[m.normTexId], uv.x, uv.y);
            normal = normalize(vec3(n.x, n.y, n.z) * 2.0f - 1.0f);
        }
        if (m.diffTexId >= 0)
        {
            float4 c = tex2D<float4>(textures[m.diffTexId], uv.x, uv.y);
            color = vec3(c.x, c.y, c.z);
        }
        if (m.roughTexId >= 0)
        {
            float4 r = tex2D<float4>(textures[m.roughTexId], uv.x, uv.y);
            metallic = glm::clamp(r.x, 0.0f, 1.0f);
            roughness = glm::clamp(r.y, 0.0f, 1.0f);
        }
    }

    vec3 diffDir = calculateRandomDirectionInHemisphere(normal, rng);
    vec3 refDir = reflect(pathSegment.ray.direction, normal);
    vec3 intersect = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;

    sample.wi.origin = intersect + normal * EPSILON;

    sample.wi.direction = glm::mix(refDir, diffDir, roughness);

    sample.lo = color;
    return sample;
}
