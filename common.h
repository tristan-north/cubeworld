#pragma once

#include <glm/vec3.hpp>

#define WIDTH 430	// screenwidth
#define HEIGHT 392	// screenheight
#define SPP  32	// samples per pixel per pass
#define RAYDEPTH 4  // needs to be at least 1 for camera ray
#define KEYBOARD_MOVESPEED 0.1f
#define EPSILON 0.0001f
#define LARGE_VAL 100000.0f
#define SKY_COLOR 0.05f, 0.06f, 0.09f
//#define SKY_COLOR 0.0f, 0.0f, 0.0f

struct Light {
	glm::vec3 pos;
	glm::vec3 dir;
	glm::vec3 color;
	float radius;
};