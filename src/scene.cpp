#include "scene.h"

#include "utilities.h"
#include "intersections.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cassert>

#include <stb_image.h>
#include <stb_image_write.h>

#define WIDTH 800
#define HEIGHT 800

using namespace std;
using json = nlohmann::json;

static int nodesUsed = 1;

static glm::mat4 convertAIMatrix(const aiMatrix4x4& m)
{
    glm::mat4 mat = glm::mat4(
        m.a1, m.b1, m.c1, m.d1,
        m.a2, m.b2, m.c2, m.d2,
        m.a3, m.b3, m.c3, m.d3,
        m.a4, m.b4, m.c4, m.d4
    );
    return mat;
}

AABB surroundingBox(AABB a, AABB b)
{
    AABB aabb;
    aabb.min = glm::vec3(fmin(a.min.x, b.min.x), fmin(a.min.y, b.min.y), fmin(a.min.z, b.min.z)) - EPSILON; // add epsilon to give bounding box leniency
    aabb.max = glm::vec3(fmax(a.max.x, b.max.x), fmax(a.max.y, b.max.y), fmax(a.max.z, b.max.z)) + EPSILON;
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
        loadAssimp(filename);
    }


}

void Scene::initCamera(const aiScene* scene)
{
    Camera& camera = state.camera;
    RenderState& state = this->state;
    
    camera.resolution.x = WIDTH;
    camera.resolution.y = HEIGHT;
    float fovy;
    state.iterations = 5000;
    state.traceDepth = 8;
    state.imageName = scene->mName.C_Str();

    if (scene->mCameras)
    {
        aiCamera* aiCam = scene->mCameras[0];

        float aspect = aiCam->mAspect;
        fovy = 2.0f * atan(tan(aiCam->mHorizontalFOV * 0.5f) / aspect);
        const auto& pos = aiCam->mPosition;
        const auto& lookat = aiCam->mLookAt;
        const auto& up = aiCam->mUp;
        camera.position = glm::vec3(pos[0], pos[1], pos[2]);
        camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
        camera.up = glm::vec3(up[0], up[1], up[2]);
    }
    else
    {
        fovy = 45.0;
        camera.position = glm::vec3(0.0, 25.0, 50.5);
        camera.lookAt = camera.position + glm::vec3(0.0, 0.0, -camera.position.z - 5.0);
        camera.up = glm::vec3(0.0, 1.0, 0.0);
    }

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
}

void Scene::convertMats(const aiScene* scene)
{

    for (int i = 0; i < scene->mNumMaterials; i++)
    {
        const aiMaterial* aim = scene->mMaterials[i];

        Material m;

        // diffuse
        aiColor3D kd(1.f, 1.f, 1.f);
        if (AI_SUCCESS == aim->Get(AI_MATKEY_COLOR_DIFFUSE, kd)) 
        {
            m.color = glm::vec3(kd.r, kd.g, kd.b);
        }

        // emissive
        aiColor3D ke(0.f, 0.f, 0.f);
        if (AI_SUCCESS == aim->Get(AI_MATKEY_COLOR_EMISSIVE, ke)) {
            m.emissive = glm::vec3(ke.r, ke.g, ke.b);
        }
        m.emittance = glm::length(m.emissive);

        // specular
        float shininess = 0.0f;
        if (AI_SUCCESS == aim->Get(AI_MATKEY_SHININESS, shininess)) {
            m.specular.color = glm::vec3(kd.r, kd.g, kd.b);
            m.specular.exponent = shininess;
            m.hasSpecular = false; //shininess > 0.0f;
        }

        // textures
        aiString texPath;
        if (aim->GetTexture(aiTextureType_BASE_COLOR, 0, &texPath) == AI_SUCCESS) 
        {
            loadTexture(scene, texPath);
            m.diffTexId = (int)texInfos.size() - 1;
        }
        if (aim->GetTexture(aiTextureType_NORMALS, 0, &texPath) == AI_SUCCESS)
        {
            loadTexture(scene, texPath);
            m.normTexId = (int)texInfos.size() - 1;
        }

        materials.push_back(m);
    }
}

void Scene::loadTexture(const aiScene* scene, aiString texPath)
{
    int w = 0, h = 0, ch = 0;
    stbi_uc* data = nullptr;

    if (texPath.data[0] == '*')
    {
        int idx = atoi(texPath.C_Str() + 1);
        const aiTexture* at = scene->mTextures[idx];

        if (at->mHeight == 0)
        {
            data = stbi_load_from_memory(
                reinterpret_cast<const stbi_uc*>(at->pcData),
                at->mWidth, &w, &h, &ch, STBI_rgb_alpha);
        }
        else
        {
            w = at->mWidth; h = at->mHeight; ch = 4;
            data = (stbi_uc*)malloc(size_t(w) * h * 4);

            for (int i = 0; i < w * h; ++i)
            {
                const aiTexel& s = at->pcData[i];
                data[4 * i + 0] = s.r;
                data[4 * i + 1] = s.g;
                data[4 * i + 2] = s.b;
                data[4 * i + 3] = s.a;
            }
        }
    }
    else
    {
        data = stbi_load(texPath.C_Str(), &w, &h, &ch, STBI_rgb_alpha);
    }

    if (!data) throw runtime_error("failed to load mesh texture");

    TexInfo info;
    info.width = w;
    info.height = h;
    info.pixels.assign(data, data + size_t(w) * h * 4);
    stbi_image_free(data);

    texInfos.push_back(info);
}

void Scene::createTextureObjects()
{
    for (size_t i = 0; i < texInfos.size(); ++i) {
        const TexInfo& info = texInfos[i];

        cudaArray_t array = nullptr;
        auto ch = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK(cudaMallocArray(&array, &ch, info.width, info.height));

        const size_t widthBytes = size_t(info.width) * 4;
        const size_t srcPitch = widthBytes;

        CUDA_CHECK(cudaMemcpy2DToArray(
            array, 0, 0,
            info.pixels.data(), srcPitch,
            widthBytes, info.height,
            cudaMemcpyHostToDevice)); // here

        cudaResourceDesc res{};
        res.resType = cudaResourceTypeArray;
        res.res.array.array = array;

        cudaTextureDesc td{};
        td.addressMode[0] = cudaAddressModeWrap;   // or Clamp
        td.addressMode[1] = cudaAddressModeWrap;
        td.filterMode = cudaFilterModeLinear;
        td.readMode = cudaReadModeNormalizedFloat; // -> [0,1]
        td.normalizedCoords = 1;

        cudaTextureObject_t tex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&tex, &res, &td, nullptr));

        textures.push_back(tex);
    }
}

void Scene::nodeDFS(const aiNode* node, const glm::mat4& parentTransform, const aiScene* scene, unordered_map<string, glm::mat4>& map)
{
    glm::mat4 local = convertAIMatrix(node->mTransformation);
    glm::mat4 world = parentTransform * local;


    for (int i = 0; i < node->mNumMeshes; i++)
    {
        int idx = node->mMeshes[i];
        std::string name = scene->mMeshes[idx]->mName.C_Str();
        map[name] = world;
    }

    for (int i = 0; i < node->mNumChildren; i++)
    {
        nodeDFS(node->mChildren[i], world, scene, map);
    }
}

void Scene::loadAssimp(const std::string& path)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate |
        aiProcess_GlobalScale |
        aiProcess_GenSmoothNormals |
        aiProcess_FlipUVs |
        aiProcess_JoinIdenticalVertices);

    cout << path << endl;

    if (!scene) {
        std::cerr << "Assimp error: " << importer.GetErrorString() << std::endl;
        throw std::runtime_error("failed to load scene");
    }

    convertMats(scene); // convert ai mats to our custom mat structs

    unordered_map<string, glm::mat4> nodeTransforms; // get node transforms for global coordinates

    nodeDFS(scene->mRootNode, glm::mat4(1.0f), scene, nodeTransforms);

    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        const aiMesh* currMesh = scene->mMeshes[i];
        glm::mat4 transform = nodeTransforms[currMesh->mName.C_Str()];

        const int baseVertex = static_cast<int>(vertices.size());
        const int matId = currMesh->mMaterialIndex;

        // find vertices
        for (unsigned int j = 0; j < currMesh->mNumVertices; j++)
        {
            Vertex vertex;
            vertex.position = glm::vec3(transform * glm::vec4(currMesh->mVertices[j].x, currMesh->mVertices[j].y, currMesh->mVertices[j].z, 1.0f));
            vertex.normal = currMesh->HasNormals() ? glm::vec3(glm::transpose(glm::inverse(transform)) * glm::vec4(currMesh->mNormals[j].x, currMesh->mNormals[j].y, currMesh->mNormals[j].z, 0.0f)) : glm::vec3(0.0f);
            vertex.texCoord = currMesh->HasTextureCoords(0) ? glm::vec2(currMesh->mTextureCoords[0][j].x, currMesh->mTextureCoords[0][j].y) : glm::vec2(0.0f);

            vertices.push_back(vertex);
        }

        for (unsigned int j = 0; j < currMesh->mNumFaces; j++)
        {
            const aiFace* face = &currMesh->mFaces[j];

            Triangle tri;
            for (unsigned int k = 0; k < face->mNumIndices; k++)
            {
                tri.vertices[k] = vertices[baseVertex + face->mIndices[k]];
            }
            tri.materialId = matId;
            triangles.push_back(tri);
        }
    }

    // BUILD ACCELERATION STRCTURES
    nodes.resize(triangles.size() * 2 - 1);
    nodes[0].left = nodes[0].right = -1;
    nodes[0].startGeom = 0;
    nodes[0].numGeoms = triangles.size();
    updateNodeAABB(0);
    subdivide(0);

    num_nodes = nodesUsed;

    // CAMERA
    initCamera(scene);
}

void Scene::updateNodeAABB(int index)
{
    BVHNode& node = nodes[index];
    node.aabb.min = glm::vec3(1e30f);
    node.aabb.max = glm::vec3(-1e30f);

    for (int i = node.startGeom; i < node.startGeom + node.numGeoms; i++)
    {
        node.aabb = surroundingBox(node.aabb, triangles[i].getAABB());
    }
}

static int axis;
int comp(const void* a, const void* b)
{
    const Triangle* at = static_cast<const Triangle*>(a);
    const Triangle* bt = static_cast<const Triangle*>(b);

    float ca = at->getCenter()[axis];
    float cb = bt->getCenter()[axis];
    
    if (ca < cb) return -1;
    if (ca > cb) return 1;
    return 0;
}

void Scene::subdivide(int index)
{
    BVHNode& node = nodes[index];

    if (node.numGeoms <= 2)
    {
        node.left = node.right = -1;
        return;
    }
    // determine split axis
    glm::vec3 extent = node.aabb.max - node.aabb.min;
    axis = 0;
    if (extent.y > extent.x) axis = 1;
    if (extent.z > extent[axis]) axis = 2;

    qsort(&triangles[nodes[index].startGeom], nodes[index].numGeoms, sizeof(Triangle), comp); // sort by axis

    int left = nodesUsed++;
    int right = nodesUsed++;

    assert(left != index && right != index);
    assert(right < (int)nodes.size());

    node.left = left;
    node.right = right;

    nodes[left].startGeom = nodes[index].startGeom;
    nodes[left].numGeoms = nodes[index].numGeoms / 2;

    nodes[right].startGeom = nodes[index].startGeom + nodes[index].numGeoms / 2;
    nodes[right].numGeoms = nodes[index].numGeoms - (nodes[index].numGeoms / 2);

    nodes[left].left = nodes[left].right = -1;
    nodes[right].left = nodes[right].right = -1;

    updateNodeAABB(left);
    updateNodeAABB(right);
    subdivide(left);
    subdivide(right);

    nodes[index].numGeoms = 0;
}



///// LEGACY JSON IMPORTING //////

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
    nodes.resize(geoms.size() * 2 - 1);
    nodes[0].left = nodes[0].right = -1;
    nodes[0].startGeom = 0;
    nodes[0].numGeoms = geoms.size();
    JSONUpdateNodeAABB(0);
    JSONSubdivide(0);
}

// JSON

void Scene::JSONUpdateNodeAABB(int index)
{
    BVHNode& node = nodes[index];
    node.aabb.min = glm::vec3(1e30f);
    node.aabb.max = glm::vec3(-1e30f);

    for (int i = node.startGeom; i < node.startGeom + node.numGeoms; i++)
    {
        node.aabb = surroundingBox(node.aabb, geoms[i].aabb);
    }
}

void Scene::JSONSubdivide(int index)
{
    BVHNode& node = nodes[index];

    if (node.numGeoms <= 2)
    {
        nodes[index].left = nodes[index].right = -1;
        return;
    }
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
    if (leftCount == 0 || leftCount == node.numGeoms)
    {
        node.left = node.right = -1;
        return;
    }

    int left = nodesUsed++;
    int right = nodesUsed++;

    nodes[left].startGeom = nodes[index].startGeom;
    nodes[left].numGeoms = leftCount;
    nodes[right].startGeom = i;
    nodes[right].numGeoms = nodes[index].numGeoms - leftCount;
    nodes[index].left = left;
    nodes[index].right = right;
    nodes[index].numGeoms = 0;
    JSONUpdateNodeAABB(left);
    JSONUpdateNodeAABB(right);
    JSONSubdivide(left);
    JSONSubdivide(right);
}