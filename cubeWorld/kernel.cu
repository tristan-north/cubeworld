#include "cutil_math.h"  // required for float3 vector math
#include <Windows.h>  // This needs to be included before openGL stuff
#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_gl_interop.h>


#include "common.h"
#include "kernel.cuh"
#include "camera.h"

#define M_PI 3.14159265359f

// output buffer
float3 *g_rgbBuffer;

struct Ray {
	float3 orig;	// ray origin
	float3 dir;		// ray direction	
	__device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

// required to convert colour to a format that OpenGL can display  
union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

// helper functions
inline __device__ float3 minf3(float3 a, float3 b){ return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
inline __device__ float3 maxf3(float3 a, float3 b){ return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
inline __device__ float minf1(float a, float b){ return a < b ? a : b; }
inline __device__ float maxf1(float a, float b){ return a > b ? a : b; }

__constant__ float aspectRatio = float(WIDTH) / HEIGHT;
__constant__ float3 gridMin = { -50.0f, 0.0f, -50.0f };
__constant__ float3 gridMax = { 50.0f, 100.0f, 50.0f };
__constant__ uint gridRes = 5;
__constant__ uint grid[5][5][5] = { 
{ { 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0 }, { 1, 0, 1, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },
{ { 1, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },
{ { 1, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 1, 0, 0, 0 } },
{ { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, 1, 1, 0, 0 } },
{ { 1, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1 }, { 1, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 } },
};

// Returns distance to intersection or 0 if no intersection
__device__ float box_intersect(const Ray &r, const float3 min, const float3 max) {

	if (fabs(r.dir.x) < EPSILON) return 0;
	if (fabs(r.dir.y) < EPSILON) return 0;
	if (fabs(r.dir.z) < EPSILON) return 0;

	float3 tmin = (min - r.orig) / r.dir;
	float3 tmax = (max - r.orig) / r.dir;

	float3 real_min = minf3(tmin, tmax);
	float3 real_max = maxf3(tmin, tmax);

	float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
	float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);

	if (minmax >= maxmin) { return maxmin > EPSILON ? maxmin : 0; }
	else return 0;

}

__device__ bool ground_intersect(const Ray &ray, float &t, float3 &color, float3 &normal) {
	float denom = dot(make_float3(0.0f, 1.0f, 0.0f), ray.dir);
	if (denom < EPSILON) {
		float3 p0l0 = make_float3(0.0f, 0.0f, 0.0f) - ray.orig;
		t = dot(p0l0, make_float3(0.0f, 1.0f, 0.0f)) / denom;
		color = { 1.0f, 1.0f, 1.0f };
		normal = { 0.0f, 1.0f, 0.0f };
		return (t >= 0);
	}
	return false;
}


__device__ inline bool grid_intersect(const Ray &ray, float &t, float3 &color, float3 &normal) {
	const float3 cellSize = { (gridMax.x - gridMin.x) / gridRes, (gridMax.y - gridMin.y) / gridRes, (gridMax.z - gridMin.z) / gridRes };

	float bboxIsecDist = 0;
	// Check if ray starts inside bbox.
	if (! ((ray.orig.x > gridMin.x && ray.orig.x < gridMax.x) && (ray.orig.y > gridMin.y && ray.orig.y < gridMax.y) && (ray.orig.z > gridMin.z && ray.orig.z < gridMax.z))) {
		// If not inside find if and where ray hits grid bbox
		bboxIsecDist = box_intersect(ray, gridMin, gridMax);
		// If misses grid entirely intersect ground plane
		if (bboxIsecDist == 0) {
			return ground_intersect(ray, t, color, normal);
		}
	}
	const float3 gridIsecPoint = ray.orig + ray.dir*bboxIsecDist;

	// rayOGridspace is the ray origin position relative to the grid origin position. Ie rayOGrid is in "grid space".
	const float3 rayOGridspace = { gridIsecPoint.x - gridMin.x, gridIsecPoint.y - gridMin.y, gridIsecPoint.z - gridMin.z };
	// This is the ray origin position in "cell space". Ie if rayOCell.x is 2.5,
	// the ray starts in the middle of the 3rd cell in x.
	const float3 rayOCellspace = { rayOGridspace.x / cellSize.x, rayOGridspace.y / cellSize.y, rayOGridspace.z / cellSize.z };

	uint3 cellIndex;
	cellIndex.x = floor(rayOCellspace.x); cellIndex.x = clamp(cellIndex.x, uint(0), gridRes - 1);
	cellIndex.y = floor(rayOCellspace.y); cellIndex.y = clamp(cellIndex.y, uint(0), gridRes - 1);
	cellIndex.z = floor(rayOCellspace.z); cellIndex.z = clamp(cellIndex.z, uint(0), gridRes - 1);
	//color = make_float3(float(cellIndex.x) / (gridRes-1), float(cellIndex.y) / (gridRes-1), float(cellIndex.z) / (gridRes-1));  return true;

	// deltaT is the distance between cell border intersections for each axis
	const float deltaTx = fabs(cellSize.x / ray.dir.x);
	const float deltaTy = fabs(cellSize.y / ray.dir.y);
	const float deltaTz = fabs(cellSize.z / ray.dir.z);

	// tx, ty and tz are how far along the ray needs to be travelled to get to the
	// next (based on current t) cell in x, next cell in y and next cell in z.
	// Whichever is smallest will be the next intersection.
	float tx = ((cellIndex.x + 1) * cellSize.x - rayOGridspace.x) / ray.dir.x;
	if (ray.dir.x < 0)
		tx = (cellIndex.x * cellSize.x - rayOGridspace.x) / ray.dir.x;
	float ty = ((cellIndex.y + 1) * cellSize.y - rayOGridspace.y) / ray.dir.y;
	if (ray.dir.y < 0)
		ty = (cellIndex.y * cellSize.y - rayOGridspace.y) / ray.dir.y;
	float tz = ((cellIndex.z + 1) * cellSize.z - rayOGridspace.z) / ray.dir.z;
	if (ray.dir.z < 0)
		tz = (cellIndex.z * cellSize.z - rayOGridspace.z) / ray.dir.z;

	// Used to either increment or decrement cell index based on if ray direction is + or -.
	int stepX = ray.dir.x > 0 ? 1 : -1;
	int stepY = ray.dir.y > 0 ? 1 : -1;
	int stepZ = ray.dir.z > 0 ? 1 : -1;

	// Traverse the grid.
	t = 0;
	bool hit = false;
	color = { 0.0f, 0.0f, 0.0f };
	const int maxCellIndex = (int)(gridRes - 1);
	normal = { 0.0f, 0.0f, 0.0f };

	while (true) {
		// Check if grid cell contents is true
		if (grid[cellIndex.x][cellIndex.y][cellIndex.z]) {
			hit = true;
			break;
		}

		// Move variables to next cell.
		if (tx <= ty && tx <= tz) {
			// tx is smallest, so we're crossing into another cell in x.
			t = tx;   // As this is the next cell boarder intersected, update t to this
			tx += deltaTx;   // update to next intersection along x
			cellIndex.x = cellIndex.x + stepX;
			normal = { 1.0f, 0.0f, 0.0f };
			normal *= -stepX;
		}
		else if (ty <= tx && ty <= tz) {
			// ty is smallest, so we're crossing into another cell in y.
			t = ty;
			ty += deltaTy;
			cellIndex.y = cellIndex.y + stepY;
			normal = { 0.0f, 1.0f, 0.0f };
			normal *= -stepY;
		}
		else if (tz <= tx && tz <= ty) {
			// tz is smallest, so we're crossing into another cell in z.
			t = tz;
			tz += deltaTz;
			cellIndex.z = cellIndex.z + stepZ;
			normal = { 0.0f, 0.0f, 1.0f };
			normal *= -stepZ;
		}

		// Break if the next cell is outside the grid.
		if (cellIndex.x > maxCellIndex || cellIndex.y > maxCellIndex || cellIndex.z > maxCellIndex) {
			t = 0;
			break;
		}
		if (cellIndex.x < 0 || cellIndex.y < 0 || cellIndex.z < 0) {
			t = 0;
			break;
		}
	}

	if (!hit) return ground_intersect(ray, t, color, normal);

	// If the normal is still 0 here, it means the ray hit the outside of the grid so need
	// to do something else to get the normal.
	if (normal.x == 0.0f && normal.y == 0.0f && normal.z == 0.0f) {
		const float epsilon = 0.001f;

		if (fabs(gridMin.x - gridIsecPoint.x) < epsilon) normal = { -1.0f, 0.0f, 0.0f };
		else if (fabs(gridMax.x - gridIsecPoint.x) < epsilon) normal = { 1.0f, 0.0f, 0.0f };
		else if (fabs(gridMin.y - gridIsecPoint.y) < epsilon) normal = { 0.0f, -1.0f, 0.0f };
		else if (fabs(gridMax.y - gridIsecPoint.y) < epsilon) normal = { 0.0f, 1.0f, 0.0f };
		else if (fabs(gridMin.z - gridIsecPoint.z) < epsilon) normal = { 0.0f, 0.0f, -1.0f };
		else normal = { 0.0f, 0.0f, 1.0f };
	}

	t += bboxIsecDist;
	color = { 0.9, 0.9, 0.9 };
	return hit;
}


// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float3 radiance(const Ray &camRay, curandState &randstate){ // returns ray color

	// colour mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	// accumulated colour
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);

	Ray r = camRay;

	for (int bounces = 0; bounces < RAYDEPTH; bounces++){  // iteration up to 4 bounces (instead of recursion in CPU code)

		// reset scene intersection function parameters
		float t = 100000; // distance to intersection 
		//float3 albedo;  // primitive colour
		//float3 emission; // primitive emission colour
		//float3 x; // intersection point
		//float3 n; // normal
		//float3 nl; // oriented normal
		//float3 d; // ray direction of next path segment

		// intersect ray with scene
		// intersect_scene keeps track of closest intersected primitive and distance to closest intersection point
		float3 color, normal;
		if (!grid_intersect(r, t, color, normal)) {
			return make_float3(1.0f, 1.0f, 1.0f) * mask; // if miss, return
		}

		/*
		Box &box = boxes[box_id];
		x = r.orig + r.dir*t;  // intersection point on object
		n = normalize(box.normalAt(x)); // normal
		nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal
		albedo = box.col;  // box colour
		emission = box.emi; // box emission
		accucolor += (mask * emission);
		*/

		// Diffuse shading
		// Create new cosine weighted ray
		// create 2 random numbers
		float r1 = 2 * M_PI * curand_uniform(&randstate);
		float r2 = curand_uniform(&randstate) * 0.99999f;  // This gets used in a sqrt and if it's 1 will lead to sqrt(0)
		float r2s = sqrtf(r2);

		// compute orthonormal coordinate frame uvw with hitpoint as origin 
		float3 w = normal;
		float3 u = normalize(cross((fabs(w.x) > .1f ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
		float3 v = cross(w, u);
		
		r.orig = r.orig + r.dir*t;
		r.orig += normal * 0.03;  // offset origin next path segment to prevent self intersection

		// compute cosine weighted random ray direction on hemisphere 
		r.dir = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

		mask *= color;
	}

	// add radiance up to a certain ray depth
	// return accumulated ray colour after all bounces are computed
	return  make_float3(0.0f, 0.0f, 0.0f);
}


__global__ void render_kernel(float3 *output, uint hashedpassnumber, float3 camOrig, float3 camDir, float3 camUp){

	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= WIDTH || y >= HEIGHT) return;

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedpassnumber + threadId, 0, 0, &randState);

	float3 pixelcol = { 0.0f, 0.0f, 0.0f }; // final pixel color     
	
	for (int s = 0; s < SPP; s++){
		float3 d = camOrig + camDir * 1.5;  // Move point out from cam origin along cam dir. This distance controls the FOV.
		float rand1 = curand_uniform(&randState) - 0.5f;
		float rand2 = curand_uniform(&randState) - 0.5f;

		d += (cross(camDir, camUp) * ((x + rand1) / WIDTH - 0.5f) * aspectRatio);  // Move our ray origin along right vector an amount depending on x value of pixel
		d += (camUp* ((y + rand2) / HEIGHT - 0.5f));  // Move our ray origin along up vector an amount depending on y value of pixel
		d = normalize(d - camOrig);  // Get vector between new point and cam orig.

		pixelcol += radiance(Ray(camOrig, d), randState)*(1. / SPP);
	}

	Colour fcolour;
	float3 colour = make_float3(clamp(pixelcol.x, 0.0f, 1.0f), clamp(pixelcol.y, 0.0f, 1.0f), clamp(pixelcol.z, 0.0f, 1.0f));
	// convert from 96-bit to 24-bit colour + perform gamma correction
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255), (unsigned char)(powf(colour.y, 1 / 2.2f) * 255), (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);
	// store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
	int i = (HEIGHT - y - 1)*WIDTH + x; // pixel index
	output[i] = make_float3(x, y, fcolour.c);
}


void cudaInit(GLuint vbo) {

	//register VBO with CUDA
	cudaGLRegisterBufferObject(vbo);
}

void cudaCleanup() {
	cudaFree(g_rgbBuffer);
}

void launchKernel(GLuint vbo, uint rand, Camera* cam) {
	// map vertex buffer object for access by CUDA 
	cudaGLMapBufferObject((void**)&g_rgbBuffer, vbo);

	dim3 blockSize(16, 16, 1);
	dim3 gridSize((int)ceil((float)WIDTH / blockSize.x), (int)ceil((float)HEIGHT / blockSize.y));

	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// launch CUDA path tracing kernel, pass in a hashed seed based on number of passes
	render_kernel << < gridSize, blockSize >> >(g_rgbBuffer, rand,
												make_float3(cam->position().x, cam->position().y, cam->position().z),
												make_float3(cam->forward().x, cam->forward().y, cam->forward().z),
												make_float3(cam->up().x, cam->up().y, cam->up().z));

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	//printf("Kernel time:  %.3f ms \n", time);

	cudaThreadSynchronize();

	// unmap buffer
	cudaGLUnmapBufferObject(vbo);
}