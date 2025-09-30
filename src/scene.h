#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);

    void buildBVH(int index);

    void updateNodeAABB(int index);

    void subdivide(int index);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<BVHNode> nodes;
    std::vector<Material> materials;
    RenderState state;
};
