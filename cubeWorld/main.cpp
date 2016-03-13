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
#include "camera.h"
#include "timer.h"

uint g_passNum = 0;
Camera g_cam;
Light g_light;

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

	launchKernel(g_vbo, WangHash(g_passNum), &g_cam, &g_light);

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

	g_window = SDL_CreateWindow("Cube World", 600, 300, WIDTH, HEIGHT, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
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

	// Set camera start position
	g_cam.setPosition(glm::vec3(0, 50, 220));
	g_cam.setViewportAspectRatio(float(WIDTH) / HEIGHT);
	
	g_light.dir = { 0.0f, -1.0f, 0.0f };
	g_light.pos = { 0.0f, 60.0f, 0.0f };
	//g_light.color = { 1.0f, 1.0f, 1.0f };
	g_light.color = { 0.1f, 0.1f, 0.1f };
	g_light.radius = 5.0f;

	return true;
}

int main(int argc, char** argv){

	init();
	cudaInit(g_vbo);

	// Event loop
	bool quit = false;
	SDL_Event e;
	int mouseCenter[2] = { 0, 0 };
	timeVal renderStartTime;
	renderStartTime = timer::getStartTime();

	SDL_StartTextInput();
	while (!quit)
	{
		while (SDL_PollEvent(&e) != 0) {
			if (e.type == SDL_QUIT) {
				quit = true;
			}

			// Handle keydowns
			else if (e.type == SDL_KEYDOWN) {
				switch (e.key.keysym.sym) {
				case SDLK_ESCAPE:
					quit = true;
					break;
				}
			}

			// Handle mouse
			else if (e.type == SDL_MOUSEBUTTONDOWN) {  // On mouse click
				mouseCenter[0] = e.button.x;
				mouseCenter[1] = e.button.y;

				SDL_SetRelativeMouseMode(SDL_TRUE);
			}
			else if (e.type == SDL_MOUSEBUTTONUP) {  // On mouse up
				SDL_SetRelativeMouseMode(SDL_FALSE);
				SDL_WarpMouseInWindow(g_window, mouseCenter[0], mouseCenter[1]);
			}
			else if (e.type == SDL_MOUSEMOTION) {
				if (e.motion.state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
					g_cam.offsetOrientation(0, 0.3f * e.motion.xrel);
					g_cam.offsetOrientation(0.3f * e.motion.yrel, 0);
				}
				else if (e.motion.state & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
					g_light.pos += glm::vec3(e.motion.xrel * KEYBOARD_MOVESPEED * 0.4f, 0.0f, 0.0f);
					g_light.pos += glm::vec3(0.0f, -e.motion.yrel * KEYBOARD_MOVESPEED * 0.4f, 0.0f);
				}
			}
		}

		// Handle keyboard movement
		float frameTime = timer::getElapsedInMs(renderStartTime);
		const Uint8* keyState = SDL_GetKeyboardState(NULL);
		if (keyState[SDL_SCANCODE_A])
			g_cam.offsetPosition(-g_cam.right() * KEYBOARD_MOVESPEED * frameTime);
		if (keyState[SDL_SCANCODE_D])
			g_cam.offsetPosition(g_cam.right() * KEYBOARD_MOVESPEED * frameTime);
		if (keyState[SDL_SCANCODE_W])
			g_cam.offsetPosition(g_cam.forward() * KEYBOARD_MOVESPEED * frameTime);
		if (keyState[SDL_SCANCODE_S])
			g_cam.offsetPosition(-g_cam.forward() * KEYBOARD_MOVESPEED * frameTime);
		if (keyState[SDL_SCANCODE_E])
			g_cam.offsetPosition(g_cam.up() * KEYBOARD_MOVESPEED * frameTime);
		if (keyState[SDL_SCANCODE_Q])
			g_cam.offsetPosition(-g_cam.up() * KEYBOARD_MOVESPEED * frameTime);

		//if (keyState[SDL_SCANCODE_KP_4])
		//	g_light.pos += glm::vec3(-1.0f, 0.0f, 0.0f) * KEYBOARD_MOVESPEED * 0.2f * frameTime;
		//if (keyState[SDL_SCANCODE_KP_6])
		//	g_light.pos += glm::vec3(1.0f, 0.0f, 0.0f) * KEYBOARD_MOVESPEED * 0.2f * frameTime;
		//if (keyState[SDL_SCANCODE_KP_8])
		//	g_light.pos += glm::vec3(0.0f, 1.0f, 0.0f) * KEYBOARD_MOVESPEED * 0.2f * frameTime;
		//if (keyState[SDL_SCANCODE_KP_5])
		//	g_light.pos += glm::vec3(0.0f, -1.0f, 0.0f) * KEYBOARD_MOVESPEED * 0.2f * frameTime;

		renderStartTime = timer::getStartTime();
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

- Figure out why surface gets brighter than light source when close to surface.
- Create light samples based on the solid angle rather than point on a sphere like here http://graphics.pixar.com/library/PhysicallyBasedLighting/paper.pdf

- Convert to glm instead of float3

- Render text info http://www.sdltutorials.com/sdl-ttf
- Make sure am sending rays through the center of each pixel
- Use FCAT to measure frame timings http://www.geforce.co.uk/hardware/technology/fcat/technology

OPTIMISATIONS
- Render primary rays at full res and store normal and position in an array
- Loop over primary rays buffer and render seconary rays at a lower res
- Shoot secondary rays spasely and filter nearby secondary rays with the same normal
- Use secondary ray samples from previous frames

- For stereo, for each primary ray, connect the hit point back to the other eye
- Max ray distance
- Optimise ground intersection, should be able to make it faster since ground normal is always (0,1,0)
- Try using cuda faster math functions https://docs.nvidia.com/cuda/cuda-c-programming-guide/#intrinsic-functions
- Cuda optimisation webinars https://developer.nvidia.com/developer-webinars (https://www.youtube.com/watch?v=vt7Hvj4oviQ&feature=player_detailpage)
- Clamp indirect sample intensity
- Shoot more light shadow rays based on the solid angle of the light to the shading point



*/