#pragma once

#include "sceneStructs.h"
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <unordered_map>

class Scene
{
private:

    struct TexInfo
    {
        std::vector<unsigned char> pixels;
        int width;
        int height;
        int channels;
    };

    void loadFromJSON(const std::string& jsonName);

    void initCamera(const aiScene* scene);

    void convertMats(const aiScene* scene);

    void loadAssimp(const std::string& path);

    void nodeDFS(const aiNode* node, const glm::mat4& parentTransform, const aiScene* scene, std::unordered_map<std::string, glm::mat4>& map);
    
    void loadTexture(const aiScene* scene, aiString texPath);

    void updateNodeAABB(int index);
    void subdivide(int index);

    // LEGACY JSON BVH
    void JSONUpdateNodeAABB(int index);
    void JSONSubdivide(int index);
public:
    Scene(std::string filename);
    void createTextureObjects();
    void loadEnv();

    std::vector<Geom> geoms;
    std::vector<BVHNode> nodes;
    std::vector<Material> materials;
    std::vector<TexInfo> texInfos;
    std::vector<cudaTextureObject_t> textures;
    std::vector<Triangle> triangles;
    std::vector<Vertex> vertices;
    RenderState state;
    int num_nodes;

    bool hasEnv = false;
    cudaArray_t envArray;
    cudaTextureObject_t env;
};
