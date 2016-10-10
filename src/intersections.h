#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define ENABLE_MESH_BBOX 1

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(const Geom& box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(const Geom& sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ bool aabbBoxIntersect(const Ray& r, glm::vec3 min, glm::vec3 max) 
{
	float tnear = FLT_MIN;
	float tfar = FLT_MAX;

	for (int i = 0; i<3; i++) // x slab then y then z
	{
		float t0, t1;
		
		if (fabs(r.direction[i]) < EPSILON)
		{
			if (r.origin[i] < min[i] || r.origin[i] > max[i])
				return false;
			else
			{
				t0 = FLT_MIN;
				t1 = FLT_MAX;
			}
		}
		else
		{
			t0 = (min[i] - r.origin[i]) / r.direction[i];
			t1 = (max[i] - r.origin[i]) / r.direction[i];
		}

		tnear = glm::max(tnear, glm::min(t0, t1));
		tfar = glm::min(tfar, glm::max(t0, t1));
	}

	if (tfar < tnear) return false; // no intersection

	if (tfar < 0) return false; // behind origin of ray

	return true;

}

__host__ __device__ float triArea(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2)
{
	return glm::length(glm::cross(p0 - p1, p2 - p1)) * 0.5f;
}

//Returns the interpolation of the triangle's three normals based on the point inside the triangle that is given.
__host__ __device__ glm::vec3 getTriangleNormal(const glm::vec3 &point, const Vertex &v0, const Vertex &v1, const Vertex &v2)
{
	float a = triArea(v0.position, v1.position, v2.position);
	float a0 = triArea(v1.position, v2.position, point);
	float a1 = triArea(v0.position, v2.position, point);
	float a2 = triArea(v0.position, v1.position, point);
	return glm::normalize(v0.normal * a0 / a + v1.normal * a1 / a + v2.normal * a2 / a);
}


__host__ __device__ void meshTriangleIntersectionTest(const Vertex* v0p, const Geom& mesh, const Ray& local_ray,
	glm::vec3 &current_intersectionPoint_obj, glm::vec3 &current_normal_obj, bool &current_outside, float& current_min_t)
{
	auto v0 = v0p[0];
	auto v1 = v0p[1];
	auto v2 = v0p[2];
	// this normal is just for getting intersection point, may be different from rendering normal
	auto plane_normal = glm::normalize(glm::cross(v1.position - v0.position, v2.position - v1.position));

	glm::vec3 v01(v1.position - v0.position);
	glm::vec3 v12(v2.position - v1.position);
	glm::vec3 v20(v0.position - v2.position);
	
	//glm::vec3 normal(glm::cross(v01, v12));

	auto direction_dot_normal = glm::dot(local_ray.direction, plane_normal);

	if (fabs(direction_dot_normal) < EPSILON)  // parallel
	{
		return;
	}

	float t = (glm::dot(v0.position - local_ray.origin, plane_normal))
		/ direction_dot_normal;

	if (t <= 0)
	{
		return;
	}

	// intersection point of ray and plane
	glm::vec3 ipoint(local_ray.origin + t * local_ray.direction);

	// check if point in triangle
	glm::vec3 v0i(ipoint - v0.position);
	glm::vec3 v1i(ipoint - v1.position);
	glm::vec3 v2i(ipoint - v2.position);

	if (glm::dot(glm::cross(v01, v0i), plane_normal) < 0 ||
		glm::dot(glm::cross(v12, v1i), plane_normal) < 0 ||
		glm::dot(glm::cross(v20, v2i), plane_normal) < 0)
	{
		// outside of triangle
		return;
	}

	if (t < current_min_t)
	{
		current_min_t = t;
		current_outside = direction_dot_normal < 0;
		current_intersectionPoint_obj = ipoint;
		current_normal_obj = getTriangleNormal(ipoint, v0, v1, v2);
		if (!current_outside)
		{
			current_normal_obj = -current_normal_obj;
		}
	}

}
/**
* Test intersection between a ray and a transformed Mesh. 
*
* @param intersectionPoint  Output parameter for point of intersection.
* @param normal             Output parameter for surface normal.
* @param outside            Output param for whether the ray came from outside.
* @return                   Ray parameter `t` value. -1 if no intersection.
*/
__host__ __device__ float meshIntersectionTest(const Geom& mesh, Ray r,
	glm::vec3 &out_intersectionPoint, glm::vec3 &out_normal, bool &out_outside
	, const Vertex* vertex_buffer
	)
{
	Ray rt;
	rt.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	rt.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

#if ENABLE_MESH_BBOX
	if (!aabbBoxIntersect(rt, mesh.bounding_box_min, mesh.bounding_box_max))
	{
		return -1;
	}
#endif

	auto current_triangle_v0_ptr = vertex_buffer + mesh.vertices_begin_index;
	auto end = current_triangle_v0_ptr + mesh.vertices_count;

	glm::vec3 current_intersectionPoint;
	glm::vec3 current_normal;
	bool current_outside;
	float current_t = FLT_MAX;

	while (current_triangle_v0_ptr < end)
	{
		meshTriangleIntersectionTest(current_triangle_v0_ptr, mesh, rt, current_intersectionPoint, current_normal, current_outside, current_t);
		current_triangle_v0_ptr += 3;
	}
	if (current_t == FLT_MAX)
	{
		return -1;
	}

	out_intersectionPoint = multiplyMV(mesh.transform, glm::vec4(current_intersectionPoint, 1.f));
	out_normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(current_normal, 0.f)));
	out_outside = current_outside;

	return glm::length(r.origin - out_intersectionPoint);
}