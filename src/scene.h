#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

class Scene {
private:
	std::string path;
    std::ifstream fp_in;
    int loadMaterial(std::string materialid);
    int loadGeom(std::string objectid);
    int loadCamera();
	void loadMesh(const std::string & model_path, Geom & geom);
public:
    Scene(std::string filename);
    ~Scene() = default;

    std::vector<Geom> geoms;
	std::vector<Vertex> vertices;
    std::vector<Material> materials;
    RenderState state;
};
