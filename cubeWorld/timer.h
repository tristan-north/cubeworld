#pragma once

#include <Windows.h>

typedef LARGE_INTEGER timeVal;

namespace timer {

	timeVal frequency = { 0 };

	timeVal getStartTime() {
		if(!frequency.QuadPart) QueryPerformanceFrequency(&frequency);

		timeVal startTime;
		QueryPerformanceCounter(&startTime);

		return startTime;
	}

	float getElapsedInMs(const timeVal startTime) {
		timeVal endTime, elapsedMicroseconds;
		QueryPerformanceCounter(&endTime);

		elapsedMicroseconds.QuadPart = endTime.QuadPart - startTime.QuadPart;
		elapsedMicroseconds.QuadPart *= 1000000;
		elapsedMicroseconds.QuadPart /= frequency.QuadPart;

		return (float)elapsedMicroseconds.QuadPart / 1000;
	}
}