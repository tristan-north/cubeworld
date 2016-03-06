/*
*  Basic CUDA based triangle mesh path tracer.
*  For background info, see http://raytracey.blogspot.co.nz/2015/12/gpu-path-tracing-tutorial-2-interactive.html
*  Based on CUDA ray tracing code from http://cg.alexandra.dk/?p=278
*  Copyright (C) 2015  Sam Lapere
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*/

#include <SDL.h>
#include <GL/glew.h>
#include <SDL_opengl.h>
#include <gl/glu.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "kernel.cuh"
#include "common.h"

uint g_passNum = 0;

SDL_Window* g_window = NULL;
SDL_GLContext g_context;

// OpenGL vertex buffer object for real-time viewport
GLuint g_vbo;

// hash function to calculate new seed for each pass
// see http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
uint WangHash(uint a) {
	a = (a ^ 61) ^ (a >> 16);
	a = a + (a << 3);
	a = a ^ (a >> 4);
	a = a * 0x27d4eb2d;
	a = a ^ (a >> 15);
	return a;
}

void renderFrame()
{
	//printf("pass: %d   ", g_passNum);
	g_passNum++;

	//clear all pixels:
	glClear(GL_COLOR_BUFFER_BIT);

	launchKernel(g_vbo, g_passNum, WangHash(g_passNum));

	//glFlush();
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, WIDTH * HEIGHT);
	glDisableClientState(GL_VERTEX_ARRAY);

}


bool init() {
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
	{
		printf("%s - SDL could not initialize! SDL Error: %s\n", SDL_GetError());
		return false;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);

	g_window = SDL_CreateWindow("cudaTracer", 600, 300, WIDTH, HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	if (g_window == NULL) {
		printf("%s - Window could not be created! SDL Error: %s\n", SDL_GetError());
		return false;
	}

	g_context = SDL_GL_CreateContext(g_window);
	if (g_context == NULL) {
		printf("%s - OpenGL context could not be created! SDL Error: %s\n", SDL_GetError());
		return false;
	}

	glewExperimental = GL_TRUE;
	GLenum nGlewError = glewInit();
	if (nGlewError != GLEW_OK) {
		printf("%s - Error initializing GLEW! %s\n", glewGetErrorString(nGlewError));
		return false;
	}
	glGetError(); // to clear the error caused deep in GLEW

	if (SDL_GL_SetSwapInterval(0) < 0) {   // This is vsync
		printf("%s - Warning: Unable to set VSync! SDL Error: %s\n", SDL_GetError());
		return false;
	}

	// init OpenGL
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, WIDTH, 0.0, HEIGHT);
	fprintf(stderr, "OpenGL initialized \n");

	//create VBO (vertex buffer object)
	glGenBuffers(1, &g_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, g_vbo);
	//initialize VBO
	unsigned int size = WIDTH * HEIGHT * sizeof(float) * 3;  // 3 floats
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	fprintf(stderr, "VBO created  \n");

	return true;
}

int main(int argc, char** argv){

	init();
	cudaInit(g_vbo);

	// Event loop
	bool quit = false;
	SDL_Event e;

	SDL_StartTextInput();
	while (!quit)
	{
		while (SDL_PollEvent(&e) != 0) {
			if (e.type == SDL_QUIT || (e.type == SDL_KEYUP && (e.key.keysym.sym == SDLK_q || e.key.keysym.sym == SDLK_ESCAPE))) {

				quit = true;
			}
		}

		renderFrame();

		SDL_GL_SwapWindow(g_window);

	}
	SDL_StopTextInput();

	// Cleanup
	cudaCleanup();

	SDL_DestroyWindow(g_window);
	g_window = NULL;

	SDL_Quit();

	return 0;
}




/* TODO

- Implement direct light sampling
- Implement grid accel struct
- Check framerate when there are no scene objects to check overhead
- Render text info http://www.sdltutorials.com/sdl-ttf

OPTIMISATIONS
- Render primary rays at full res and store normal and position in an array
- Loop over primary rays buffer and render seconary rays at a lower res
- Shoot secondary rays spasely and filter nearby secondary rays with the same normal
- Use secondary ray samples from previous frames

- For stereo, for each primary ray, connect the hit point back to the other eye








*/