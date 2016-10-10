#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum class GeomType {
	SPHERE,
	CUBE,
	MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
	int vertices_begin_index;
	int vertices_count;
	glm::vec3 bounding_box_min; // only used for meshes atm
	glm::vec3 bounding_box_max;
};

struct Vertex 
{
	glm::vec3 position;
	glm::vec3 normal;
	// glm::vec2 uv
	Vertex(const glm::vec3& position, const glm::vec3& normal)
		: position(position)
		, normal(normal)
	{}
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment 
{
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;

    __host__ __device__ bool terminated() const
    {
        return remainingBounces < 0;
    }

    __host__ __device__ void terminate() 
    {
        remainingBounces = -1;
    }
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
	float t;
	glm::vec3 intersection_point;
	glm::vec3 surfaceNormal;
	int materialId;

	__host__ __device__ bool exists()
	{
		return t >= 0;
	}
};


