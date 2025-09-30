#include "scene.h"

#include "utilities.h"
#include "intersections.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <random>

using namespace std;
using json = nlohmann::json;

AABB surroundingBox(AABB a, AABB b)
{
    AABB aabb;
    aabb.min = glm::vec3(fmin(a.min.x, b.min.x), fmin(a.min.y, b.min.y), fmin(a.min.z, b.min.z));
    aabb.max = glm::vec3(fmax(a.max.x, b.max.x), fmax(a.max.y, b.max.y), fmax(a.max.z, b.max.z));
    return aabb;
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.exponent = p["ROUGHNESS"];
            newMaterial.hasSpecular = true;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (type == "cube")
        {
            boxAABB(newGeom);
        }
        else
        {
            sphereAABB(newGeom);
        }

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    // BUILD ACCELERATION STRCTURES
    BVHNode root;
    nodes.push_back(root);
    buildBVH(0);
}

void Scene::buildBVH(int index)
{
    BVHNode& root = nodes[index];
    root.left = root.right = -1;
    root.startGeom = 0;
    root.numGeoms = geoms.size();
    updateNodeAABB(index);
    subdivide(index);
}

void Scene::updateNodeAABB(int index)
{
    BVHNode& node = nodes[index];
    node.aabb.min = glm::vec3(1e30f);
    node.aabb.max = glm::vec3(-1e30f);

    for (int i = node.startGeom; i < node.startGeom + node.numGeoms; i++)
    {
        node.aabb = surroundingBox(node.aabb, geoms[i].aabb);
    }
}

void Scene::subdivide(int index)
{
    BVHNode& node = nodes[index];

    if (node.numGeoms <= 2) return;

    // determine split axis
    glm::vec3 extent = node.aabb.max - node.aabb.min;
    int xyz = 0;
    if (extent.y > extent.x) xyz = 1;
    if (extent.z > extent[xyz]) xyz = 2;
    float splitPos = node.aabb.min[xyz] + extent[xyz] * 0.5f;

    // split in place
    int i = node.startGeom;
    int j = i + node.numGeoms - 1;

    while (i <= j)
    {
        if (geoms[i].getCenter()[xyz] < splitPos)
        {
            i++;
        }
        else
        {
            swap(geoms[i], geoms[j--]);
        }    
    }

    int leftCount = i - node.startGeom;
    if (leftCount == 0 || leftCount == node.numGeoms) return;

    // create child nodes
    BVHNode leftNode;
    BVHNode rightNode;
    int left = nodes.size();
    nodes.push_back(leftNode);
    int right = nodes.size();
    nodes.push_back(rightNode);

    nodes[left].startGeom = node.startGeom;
    nodes[left].numGeoms = leftCount;
    nodes[right].startGeom = i;
    nodes[right].numGeoms = node.numGeoms - leftCount;
    node.left = left;
    node.right = right;
    node.numGeoms = 0;
    updateNodeAABB(left);
    updateNodeAABB(right);
    subdivide(left);
    subdivide(right);
}