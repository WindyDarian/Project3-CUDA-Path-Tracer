#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"

// TOGGLE THEM
#define ENABLE_STREAM_COMPACTION 1  // Optimization done: compact a forwarding indices array instead of dev_paths
#define SORT_PATH_BY_MATERIAL 0
// #define CACHE_FIRST_INTERSECTION 1 // now enable when STOCHASTIC_SAMPLING is not enabled
#define STOCHASTIC_SAMPLING 1

#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
		int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

Scene * hst_scene = nullptr;
glm::vec3 * dev_image = nullptr;
Geom * dev_geoms = nullptr;
Material * dev_materials = nullptr;
Vertex * dev_vertices = nullptr;
PathSegment * dev_paths = nullptr;
ShadeableIntersection * dev_intersections = nullptr;
ShadeableIntersection * dev_cached_first_intersections = nullptr;
bool first_intersection_cached = false;
int* dev_active_path_indices = nullptr;
int* dev_material_ids = nullptr; // used for sorting indices array without passing in comparsion functions (thus we can take advantage of thrust's radix sort optimization?)

PathSegment * reshuffled_dev_paths = nullptr;
ShadeableIntersection * reshuffled_dev_intersections = nullptr;

// TODO: static variables for device memory, any extra info you need, etc
// ...

void pathtraceInit(Scene *scene) {
	hst_scene = scene;
	const auto& cam = hst_scene->state.camera;
	auto pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_cached_first_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cached_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	first_intersection_cached = false;

	// TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_active_path_indices, pixelcount * sizeof(dev_active_path_indices[0]));
	cudaMemset(dev_active_path_indices, 0, pixelcount * sizeof(dev_active_path_indices[0]));
	cudaMalloc(&dev_material_ids, pixelcount * sizeof(dev_material_ids[0]));
	cudaMemset(dev_material_ids, 0, pixelcount * sizeof(dev_material_ids[0]));

	cudaMalloc(&reshuffled_dev_paths, pixelcount * sizeof(reshuffled_dev_paths[0]));
	cudaMalloc(&reshuffled_dev_intersections, pixelcount * sizeof(reshuffled_dev_intersections[0]));

	cudaMalloc(&dev_vertices, scene->vertices.size() * sizeof(dev_vertices[0]));
	cudaMemcpy(dev_vertices, scene->vertices.data(), scene->vertices.size() * sizeof(dev_vertices[0]), cudaMemcpyHostToDevice);

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	cudaFree(dev_cached_first_intersections);
	// TODO: clean up any extra device memory you created
	cudaFree(dev_active_path_indices);
	cudaFree(dev_material_ids);
	cudaFree(reshuffled_dev_paths);
	cudaFree(reshuffled_dev_intersections);

	cudaFree(dev_vertices);

	checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int* path_indices)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	// TODO: DOF

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
#if STOCHASTIC_SAMPLING
		auto rng = makeSeededRandomEngine(iter, x, y);
		thrust::uniform_real_distribution<float> u01(0, 1);
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + u01(rng) - 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + u01(rng) - 0.5f)
		);
#else
		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;

		path_indices[index] = index;
	}
}

__global__ void computeIntersections(
	int num_paths
	, const PathSegment * pathSegments
	, ShadeableIntersection * intersections
	, const Geom * geoms
	, int geoms_size
	, const Vertex * vertex_buffer
	, int* path_material_ids
	, int* dev_active_path_indices
	)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= num_paths) return;

	auto path_index = dev_active_path_indices[index];

	if (path_index < 0) return;
	
	auto pathSegment = pathSegments[path_index]; // DONE: ref or not to ref??? - no ref is faster

	if (pathSegment.terminated()) return;

	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	// naive parse through global geoms
	for (int i = 0; i < geoms_size; i++)
	{
		auto geom = geoms[i];

		if (geom.type == GeomType::CUBE)
		{
			t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == GeomType::SPHERE)
		{
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == GeomType::MESH)
		{
			t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, vertex_buffer);
		}
		// TODO: add more intersection tests here... triangle? metaball? CSG?

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t)
		{
			t_min = t;
			hit_geom_index = i;
			intersect_point = tmp_intersect;
			normal = tmp_normal;
		}
	}


	auto& intersection = intersections[path_index];
	if (hit_geom_index == -1)
	{
		intersection.t = -1.0f;
		intersection.materialId = -1; // for sorting by material
		path_material_ids[path_index] = -1;
		return;
	}

	auto material_id = geoms[hit_geom_index].materialid;
	intersection.t = t_min;
	intersection.materialId = material_id;
	intersection.surfaceNormal = normal;
	intersection.intersection_point = intersect_point;
	intersection.outside = outside;
	path_material_ids[index] = material_id;
}

__device__ void shadeAndScatter(
	  PathSegment& path_segment
	, const ShadeableIntersection& intersection
	, const Material& material
	, thrust::default_random_engine& rng
	)
{

	if (intersection.t <= 0.0f)
	{
		// If there was no intersection, color the ray black.
		// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
		// used for opacity, in which case they can indicate "no opacity".
		// This can be useful for post-processing and image compositing.
		path_segment.color = glm::vec3(0.0f);
		path_segment.terminate();
		// terminate the ray if the intersection does not exist... 

		return;
	}

	// If the material indicates that the object was a light, "light" the ray
	if (material.emittance > 0.0f)
	{
		path_segment.color *= (material.color * material.emittance);
		path_segment.terminate();
		return;
	}

	// A path which could never reach a light
	if (path_segment.remainingBounces <= 0)
	{
		path_segment.color = glm::vec3(0.0f);
		path_segment.terminate();
		return;
	}

	path_segment.remainingBounces--;
	evaluateBsdfAndScatter(
		  path_segment.ray
		, path_segment.color
		, intersection.intersection_point
		, intersection.surfaceNormal
		, intersection.outside
		, material
		, rng
	);
}

__device__ void tryGatherPath(glm::vec3 * image, const PathSegment& path_segment)
{
	if (path_segment.terminated())
	{
		// only terminated paths can contribute to color
		image[path_segment.pixelIndex] += path_segment.color;
	}
}

/**
* Gather the path if the path is terminated,
* and add the output to current iteration's output image.
*/
__global__ void kernShadeScatterAndGatherTerminated(
	int iter
	, int depth
	, int num_paths
	, PathSegment * pathSegments
	, const ShadeableIntersection * intersections
	, const Material * materials
	, glm::vec3* image
	, int* active_path_indices)
{
	auto index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= num_paths) return;
	auto path_index = active_path_indices[index];
	if (path_index < 0) return;

	auto& path_segment = pathSegments[path_index]; 
	if (path_segment.terminated()) return;
	
	auto intersection = intersections[path_index]; 
	auto material = materials[intersection.materialId];   // DONE: compare speed between ref and value, one in shadeAndScatter also
	auto rng = makeSeededRandomEngine(iter, path_index, depth); 
	shadeAndScatter(path_segment, intersection, material, rng);

	tryGatherPath(image, path_segment);

	if (path_segment.terminated())
	{
		active_path_indices[index] = -1;
	}
}

__global__ void kernReshufflePaths(
	 int num_paths
	, int* indices
	, PathSegment * path_segments
	, ShadeableIntersection * intersections
	, PathSegment * new_paths
	, ShadeableIntersection * new_intersections
	)
{
	auto path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index >= num_paths) return;

	auto old_index = indices[path_index];
	new_paths[path_index] = path_segments[old_index];
	new_intersections[path_index] = intersections[old_index];
	indices[path_index] = path_index;
}


struct isPathTerminatedForIndex
{

	__host__ __device__
		bool operator()(const int& path_index)
	{
		return path_index < 0;
	}
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera &cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	// DONE: perform one iteration of path tracing

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths, dev_active_path_indices);
	checkCUDAError("generate camera ray");

	int depth = 0;

	//thrust::device_ptr<PathSegment> thrust_dev_paths(dev_paths);
	//PathSegment* dev_path_end = dev_paths + pixelcount;
	auto num_paths = pixelcount;
	auto num_paths_active = num_paths;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete)
	{
		// no need to memset here maybe

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths_active + blockSize1d - 1) / blockSize1d;

		if (depth == 0 && first_intersection_cached)
		{
			cudaMemcpy(dev_intersections, dev_cached_first_intersections, num_paths * sizeof(dev_cached_first_intersections[0]), cudaMemcpyDeviceToDevice);
		}
		else
		{
			computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
				num_paths_active
				, dev_paths
				, dev_intersections
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_vertices
				, dev_material_ids
				, dev_active_path_indices
			);
		}
//#if CACHE_FIRST_INTERSECTION
#if !STOCHASTIC_SAMPLING
		// cache first intersection
		if (!first_intersection_cached)
		{
			first_intersection_cached = true;
			cudaMemcpy(dev_cached_first_intersections, dev_intersections, num_paths * sizeof(dev_cached_first_intersections[0]), cudaMemcpyDeviceToDevice);
		}
#endif

#if SORT_PATH_BY_MATERIAL 
		// DONE: reorder paths by material
		// DONE: sort indices only and reshuffle (deleted, see readme for link to older commit), faster than previous method but slower than without sorting 
		// DONE: sort indices only and use index array to forward query in intersection and shading stages. Faster than both methods above but still slower than without sorting
		// Conclusion: sorting is costy
		thrust::sort_by_key(thrust::device, dev_material_ids, dev_material_ids + num_paths_active, dev_active_path_indices);

#endif

		kernShadeScatterAndGatherTerminated <<<numblocksPathSegmentTracing, blockSize1d>>> (
			  iter
			, depth
			, num_paths_active
			, dev_paths
			, dev_intersections
			, dev_materials
			, dev_image
			, dev_active_path_indices
		);

#if ENABLE_STREAM_COMPACTION
		// compacting a index proxy array is faster than compacting the dev_paths on my computer
		auto new_end = thrust::remove_if(thrust::device, dev_active_path_indices, dev_active_path_indices + num_paths_active, isPathTerminatedForIndex()); 
		num_paths_active = new_end - dev_active_path_indices;
#endif
		// TODO: compact a pointer array instead to see if there is any performance increase

		depth++;

#if ENABLE_STREAM_COMPACTION
		iterationComplete = num_paths_active <= 0; 
#else
		iterationComplete = num_paths_active <= 0 || depth > traceDepth;  // safest one
#endif
		
	}

	//// Assemble this iteration and apply it to the image
	//dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	//finalGather <<<numBlocksPixels, blockSize1d >>> (num_paths, dev_image, dev_paths);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO <<<blocksPerGrid2d, blockSize2d >>> (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
