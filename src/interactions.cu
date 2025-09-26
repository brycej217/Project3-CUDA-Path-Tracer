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

__host__ __device__ Sample calculateSpecular(
    PathSegment& pathSegment,
    ShadeableIntersection& intersection,
    const Material& m,
    thrust::default_random_engine& rng)
{
    Sample sample;

    vec3 intersect = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;
    sample.wi.origin = intersect + intersection.surfaceNormal * EPSILON;
    sample.wi.direction = reflect(pathSegment.ray.direction, intersection.surfaceNormal);
    sample.lo = m.specular.color;

    return sample;
}

__host__ __device__ Sample calculateDiffuse(
    PathSegment& pathSegment,
    ShadeableIntersection& intersection,
    const Material& m,
    thrust::default_random_engine& rng)
{
    Sample sample;

    vec3 intersect = pathSegment.ray.origin + pathSegment.ray.direction * intersection.t;
    sample.wi.origin = intersect + intersection.surfaceNormal * EPSILON;
    sample.wi.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
    sample.lo = m.color; // attenuate color (perfectly diffuse surface scatter light equally in all directions so amount of potential light coming towards wo is constant)

    return sample;
}

__host__ __device__ Sample sampleBSDF(
    PathSegment& pathSegment,
    ShadeableIntersection& intersection,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // diffuse
    if (m.hasSpecular)
    {
        return calculateSpecular(pathSegment, intersection, m, rng);
    }

    return calculateDiffuse(pathSegment, intersection, m, rng);
}
