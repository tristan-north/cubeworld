#pragma once

typedef unsigned int uint;

void cudaInit(GLuint vbo);
void cudaCleanup();
void launchKernel(GLuint vbo, uint passNum, uint rand);
