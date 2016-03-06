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

#define M_PI 3.14159265359f

// hardcoded camera position
__device__ float3 g_firstcamorig = { 50, 52, 295.6 };

// output buffer
float3 *g_rgbBuffer;

// buffer for accumulating samples over several passes
float3* accumulatebuffer;

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


// SPHERES

struct Sphere {

	float rad;				// radius 
	float3 pos, emi, col;	// position, emission, color 

	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 

		// Ray/sphere intersection
		// Quadratic formula required to solve ax^2 + bx + c = 0 
		// Solution x = (-b +- sqrt(b*b - 4ac)) / 2a
		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 

		float3 op = pos - r.orig;  // 
		float t, epsilon = 0.01f;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad; // discriminant
		if (disc<0) return 0; else disc = sqrtf(disc);
		return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0);
	}
};


// AXIS ALIGNED BOXES

// helper functions
inline __device__ float3 minf3(float3 a, float3 b){ return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
inline __device__ float3 maxf3(float3 a, float3 b){ return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
inline __device__ float minf1(float a, float b){ return a < b ? a : b; }
inline __device__ float maxf1(float a, float b){ return a > b ? a : b; }

struct Box {

	float3 min; // minimum bounds
	float3 max; // maximum bounds
	float3 emi; // emission
	float3 col; // colour

	// ray/box intersection
	// for theoretical background of the algorithm see 
	// http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
	// optimised code from http://www.gamedev.net/topic/495636-raybox-collision-intersection-point/
	__device__ float intersect(const Ray &r) const {

		float epsilon = 0.001f; // required to prevent self intersection

		float3 tmin = (min - r.orig) / r.dir;
		float3 tmax = (max - r.orig) / r.dir;

		float3 real_min = minf3(tmin, tmax);
		float3 real_max = maxf3(tmin, tmax);

		float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
		float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);

		if (minmax >= maxmin) { return maxmin > epsilon ? maxmin : 0; }
		else return 0;

	}

	// calculate normal for point on axis aligned box
	__device__ float3 Box::normalAt(float3 &point) {

		float3 normal = make_float3(0.f, 0.f, 0.f);
		float epsilon = 0.001f;

		if (fabs(min.x - point.x) < epsilon) normal = make_float3(-1, 0, 0);
		else if (fabs(max.x - point.x) < epsilon) normal = make_float3(1, 0, 0);
		else if (fabs(min.y - point.y) < epsilon) normal = make_float3(0, -1, 0);
		else if (fabs(max.y - point.y) < epsilon) normal = make_float3(0, 1, 0);
		else if (fabs(min.z - point.z) < epsilon) normal = make_float3(0, 0, -1);
		else normal = make_float3(0, 0, 1);

		return normal;
	}
};

// scene: 9 spheres forming a Cornell box
// small enough to fit in constant GPU memory
__constant__ Sphere spheres[] = {
	// FORMAT: { float radius, float3 position, float3 emission, float3 colour }
	// cornell box
	{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, .01f, .01f } }, //Left
	{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .01f, 1.0f, .01f } }, //Right 
	{ 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Back 
	{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 0.00f, 0.00f, 0.00f } }, //Front 
	{ 1e5f, { 50.0f, -1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Bottom 
	{ 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Top 
	{ 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 0.99f, 0.99f, 0.99f } }, // small sphere 1
	{ 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.f, .0f }, { 0.09f, 0.49f, 0.3f } }, // small sphere 2
	{ 600.0f, { 50.0f, 681.6f - .5f, 81.6f }, { 10.0f, 7.0f, 5.0f }, { 0.0f, 0.0f, 0.0f } }  // Light 12, 10 ,8

	//outdoor scene: radius, position, emission, color, material

	//{ 1600, { 3000.0f, 10, 6000 }, { 37, 34, 30 }, { 0.f, 0.f, 0.f } },  // 37, 34, 30 // sun
	//{ 1560, { 3500.0f, 0, 7000 }, { 50, 25, 2.5 }, { 0.f, 0.f, 0.f } },  //  150, 75, 7.5 // sun 2
	//{ 10000, { 50.0f, 40.8f, -1060 }, { 0.0003, 0.01, 0.15 }, { 0.175f, 0.175f, 0.25f } }, // sky
	//{ 100000, { 50.0f, -100000, 0 }, { 0.0, 0.0, 0 }, { 0.8f, 0.2f, 0.f } }, // ground
	//{ 110000, { 50.0f, -110048.5, 0 }, { 3.6, 2.0, 0.2 }, { 0.f, 0.f, 0.f } },  // horizon brightener
	//{ 4e4, { 50.0f, -4e4 - 30, -3000 }, { 0, 0, 0 }, { 0.2f, 0.2f, 0.2f } }, // mountains
	//{ 82.5, { 30.0f, 180.5, 42 }, { 16, 12, 6 }, { .6f, .6f, 0.6f } },  // small sphere 1
	//{ 12, { 115.0f, 10, 105 }, { 0.0, 0.0, 0.0 }, { 0.9f, 0.9f, 0.9f } },  // small sphere 2
	//{ 22, { 65.0f, 22, 24 }, { 0, 0, 0 }, { 0.9f, 0.9f, 0.9f } }, // small sphere 3
};

__constant__ Box boxes[] = {
	// FORMAT: { float3 minbounds,    float3 maxbounds,         float3 emission,    float3 colour }
	{ { 5.0f, 0.0f, 70.0f }, { 45.0f, 11.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },
	{ { 85.0f, 0.0f, 95.0f }, { 95.0f, 20.0f, 105.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },
	{ { 75.0f, 20.0f, 85.0f }, { 105.0f, 22.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },

	{ { 75.0f, 25.0f, 85.0f }, { 105.0f, 27.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },
	{ { 75.0f, 30.0f, 85.0f }, { 105.0f, 32.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },
	{ { 75.0f, 35.0f, 85.0f }, { 105.0f, 37.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },
	{ { 75.0f, 40.0f, 85.0f }, { 105.0f, 42.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },

	{ { 15.0f, 25.0f, 85.0f }, { 30.0f, 27.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },
	{ { 15.0f, 30.0f, 85.0f }, { 30.0f, 32.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },
	{ { 15.0f, 35.0f, 85.0f }, { 30.0f, 37.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },
	{ { 15.0f, 40.0f, 85.0f }, { 30.0f, 42.0f, 115.0f }, { .0f, .0f, 0.0f }, { 0.5f, 0.5f, 0.5f } },
};


__device__ inline bool intersect_scene(const Ray &r, float &t, int &sphere_id, int &box_id, int &geomtype){

	float d = 1e21;
	float k = 1e21;
	float inf = t = 1e20;

	// SPHERES
	// intersect all spheres in the scene
	float numspheres = sizeof(spheres) / sizeof(Sphere);
	for (int i = int(numspheres); i--;)  // for all spheres in scene
		// keep track of distance from origin to closest intersection point
		if ((d = spheres[i].intersect(r)) && d < t){ t = d; sphere_id = i; geomtype = 1; }

	// BOXES
	// intersect all boxes in the scene
	float numboxes = sizeof(boxes) / sizeof(Box);
	for (int i = int(numboxes); i--;) // for all boxes in scene
		if ((k = boxes[i].intersect(r)) && k < t){ t = k; box_id = i; geomtype = 2; }

	// t is distance to closest intersection of ray with all primitives in the scene (spheres, boxes and triangles)
	return t<inf;
}

// radiance function
// compute path bounces in scene and accumulate returned color from each path sgment
__device__ float3 radiance(Ray &r, curandState &randstate){ // returns ray color

	// colour mask
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);
	// accumulated colour
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f);

	for (int bounces = 0; bounces < RAYDEPTH; bounces++){  // iteration up to 4 bounces (instead of recursion in CPU code)

		// reset scene intersection function parameters
		float t = 100000; // distance to intersection 
		int sphere_id = -1;
		int box_id = -1;   // index of intersected sphere 
		int geomtype = -1;
		float3 albedo;  // primitive colour
		float3 emission; // primitive emission colour
		float3 x; // intersection point
		float3 n; // normal
		float3 nl; // oriented normal
		float3 d; // ray direction of next path segment

		// intersect ray with scene
		// intersect_scene keeps track of closest intersected primitive and distance to closest intersection point
		if (!intersect_scene(r, t, sphere_id, box_id, geomtype))
			return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black

		// else: we've got a hit with a scene primitive
		// determine geometry type of primitive: sphere/box/triangle

		// if sphere:
		if (geomtype == 1){
			Sphere &sphere = spheres[sphere_id]; // hit object with closest intersection
			x = r.orig + r.dir*t;  // intersection point on object
			n = normalize(x - sphere.pos);		// normal
			nl = dot(n, r.dir) < 0 ? n : n * -1; // correctly oriented normal
			albedo = sphere.col;   // object colour
			emission = sphere.emi;  // object emission
			accucolor += (mask * emission);
		}

		// if box:
		if (geomtype == 2){
			Box &box = boxes[box_id];
			x = r.orig + r.dir*t;  // intersection point on object
			n = normalize(box.normalAt(x)); // normal
			nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal
			albedo = box.col;  // box colour
			emission = box.emi; // box emission
			accucolor += (mask * emission);
		}


		// Diffuse shading
		// ideal diffuse reflection (see "Realistic Ray Tracing", P. Shirley)

		// create 2 random numbers
		float r1 = 2 * M_PI * curand_uniform(&randstate);
		float r2 = curand_uniform(&randstate);
		float r2s = sqrtf(r2);

		// compute orthonormal coordinate frame uvw with hitpoint as origin 
		float3 w = nl;
		float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
		float3 v = cross(w, u);

		// compute cosine weighted random ray direction on hemisphere 
		d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

		// offset origin next path segment to prevent self intersection
		x += nl * 0.03;

		mask *= albedo;


		// set up origin and direction of next path segment
		r.orig = x;
		r.dir = d;
	}

	// add radiance up to a certain ray depth
	// return accumulated ray colour after all bounces are computed
	return accucolor;
}


__global__ void render_kernel(float3 *output, float3* accumbuffer, int passnumber, uint hashedpassnumber){

	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x >= WIDTH || y >= HEIGHT) return;

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedpassnumber + threadId, 0, 0, &randState);

	Ray cam(g_firstcamorig, normalize(make_float3(0, -0.042612, -1)));
	float3 cx = make_float3(WIDTH * .5135 / HEIGHT, 0.0f, 0.0f);  // ray direction offset along X-axis 
	float3 cy = normalize(cross(cx, cam.dir)) * .5135; // ray dir offset along Y-axis, .5135 is FOV angle
	float3 pixelcol; // final pixel color       

	int i = (HEIGHT - y - 1)*WIDTH + x; // pixel index

	pixelcol = make_float3(0.0f, 0.0f, 0.0f); // reset to zero for every pixel	

	for (int s = 0; s < SPP; s++){

		// compute primary ray direction
		float3 d = cx*((.25 + x) / WIDTH - .5) + cy*((.25 + y) / HEIGHT - .5) + cam.dir;
		// normalize primary ray direction
		d = normalize(d);
		// add accumulated colour from path bounces
		pixelcol += radiance(Ray(cam.orig + d * 40, d), randState)*(1. / SPP);
	}       // Camera rays are pushed ^^^^^ forward to start in interior 

	// add pixel colour to accumulation buffer (accumulates all samples) 
	accumbuffer[i] += pixelcol;
	// averaged colour: divide colour by the number of calculated passNum so far
	float3 tempcol = accumbuffer[i] / passnumber;

	//tempcol = pixelcol;  // This stops the accumulation of samples over time

	Colour fcolour;
	float3 colour = make_float3(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));
	// convert from 96-bit to 24-bit colour + perform gamma correction
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255), (unsigned char)(powf(colour.y, 1 / 2.2f) * 255), (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);
	// store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
	output[i] = make_float3(x, y, fcolour.c);
}


void cudaInit(GLuint vbo) {
	// allocate memmory for the accumulation buffer on the GPU
	cudaMalloc(&accumulatebuffer, WIDTH * HEIGHT * sizeof(float3));

	//register VBO with CUDA
	cudaGLRegisterBufferObject(vbo);
}

void cudaCleanup() {
	cudaFree(accumulatebuffer);
	cudaFree(g_rgbBuffer);
}

void launchKernel(GLuint vbo, uint passNum, uint rand) {
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
	render_kernel << < gridSize, blockSize >> >(g_rgbBuffer, accumulatebuffer, passNum, rand);  // launches CUDA render kernel from the host

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	//printf("Kernel time:  %.3f ms \n", time);

	cudaThreadSynchronize();

	// unmap buffer
	cudaGLUnmapBufferObject(vbo);
}