#pragma once

#include <cstdlib>
#include <cstring>

namespace Matrix {

template<class T>
void generateMatrix(T* matrix, size_t rows, size_t columns);

template<class T>
void setZero(T* matrix, size_t rows, size_t columns);

template<class T>
void simpleMultiplication(T* M1, T* M2, T* res, size_t M, size_t N, size_t K);

template<class T>
bool testMultiplication(T* M1, T* M2, size_t size);

}

