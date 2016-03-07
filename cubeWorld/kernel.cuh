#pragma once

typedef unsigned int uint;

class Camera;

void cudaInit(GLuint vbo);
void cudaCleanup();
void launchKernel(GLuint vbo, uint rand, Camera* cam);
