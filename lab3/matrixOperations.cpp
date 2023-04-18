#include "matrixOperations.h"

namespace Matrix {

template<class T>
void generateMatrix(T* matrix, size_t rows, size_t columns) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < columns; j++) {
            matrix[i*columns + j] = std::rand() / 100;
        }
    }
}

template<class T>
void setZero(T* matrix, size_t rows, size_t columns) {
    std::memset(matrix, 0, rows * columns * sizeof(T));
}

template<class T>
void simpleMultiplication(T* M1, T* M2, T* res, size_t N, size_t M, size_t K) {
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < K; j++) {
            T sum = 0;
            for (size_t q = 0; q < M; q++) {
                sum += M1[i*M + q] * M2[q*K + j];
            } 
            res[i*K + j] += sum; 
        }
    }
}

template<class T>
bool testMultiplication(T* M1, T* M2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (M1[i] != M2[i]) return false;
    }
    return true;
}

}