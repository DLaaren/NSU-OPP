#include <cmath>
#include <climits>
#include <omp.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>

#define N 2500
#define EPSILON 0.00001 //10^(-5)
#define PARAMETER 0.01

double randDouble() {
    return ( (double)rand() / RAND_MAX ) * 4.0 - 2.0;
}

void generateMatrix(double *M, unsigned int columns, unsigned int rows) {
    for (unsigned int i = 0; i < columns; i++) {
        for (unsigned int j = 0; j < rows; j++) {
            M[i*rows + j] = randDouble();
            if (i == j) {
                M[i*rows + j] = fabs(M[i*rows + j]) + 124.0; //выполнение условия диагонального преобладания
            }
        }
    }
}

void mul(double *M1, double *M2, double *resM) {
    unsigned int i, j;
    #pragma omp parallel for private (j)
    // i will be divided between threads
    // and all threads will have their own copies of j
    // look in the explanation1.jpg
        for (i = 0; i < N; i++) {
            resM[i] = 0;
            for (j = 0; j < N; j++) {
                resM[i] += M1[i*N + j] * M2[j];
            }
        }
}

void scalMult(double *M, double scalar) {
    #pragma omp parallel for
    for (unsigned int i = 0; i < N; i++) {
        M[i] *= scalar;
    }
}

void sub(const double *M1, const double *M2, double *resM) {
    #pragma omp parallel for
        for (unsigned int i = 0; i < N; i++) {
            resM[i] = M1[i] - M2[i];
        }
}

double EuclideanNorm(const double *M) {
    double norm = 0;
    #pragma omp parallel for reduction (+:norm)
        for (unsigned int i = 0; i < N; i++) {
            norm += M[i]*M[i];
        }
    return sqrt(norm);
}

void copyVector(const double *M1, double *M2) {
    #pragma omp parallel for
    for (unsigned int i = 0; i < N; i++) {
        M2[i] = M1[i];
    }
}

void printVector(double *vector) {
    printf("\n");
    for (unsigned int i = 0; i < N; i++) {
        printf("%lf ", vector[i]);
    }
    printf("\n");
}

void printMatrix(double *M, unsigned int columns, unsigned int rows) {
    printf("\n");
    for (unsigned int i = 0; i < columns; i++) {
        for (unsigned int j = 0; j < rows; j++) {
            printf("%lf ", M[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    srand(time(NULL));
    
    double *A = (double*)malloc(sizeof(double) * N * N);
    generateMatrix(A, N, N);
    //printMatrix(A, N, N);

    double *b = (double*)malloc(sizeof(double) * N);
    generateMatrix(b, N, 1);

    double *prevX = (double*)malloc(sizeof(double) * N);
    generateMatrix(prevX, N, 1);

    double *Ax = (double*)malloc(sizeof(double) * N);
    double *nextX = (double*)malloc(sizeof(double) * N);
    copyVector(prevX, nextX);
    double tau = PARAMETER;
    double result = INT_MAX;
    double lastResult = INT_MAX;
    unsigned int itCount = 0;
    double norm_b = EuclideanNorm(b); // ||b||

    omp_set_num_threads(4); 
    double start = omp_get_wtime();
    while (result > EPSILON) {
        copyVector(nextX, prevX);

        mul(A, prevX, Ax); // Ax = A * prevX; 
        sub(Ax, b, Ax); // Ax - b

        double norm_Ax = EuclideanNorm(Ax); // ||Ax - b||

        scalMult(Ax, tau); // tau * (Ax - b)
        sub(prevX, Ax, nextX); // x - tau * (Ax - b)

        result = norm_Ax / norm_b; //  ||Ax - b|| / ||b||
        itCount++;
        if ((itCount >= 1000 && lastResult > result) || result == INFINITY) {
            if (tau < 0) {
                printf("Doesn't have solution\n");
                result = 0;
            } else {
                tau = -PARAMETER;
                result = INT_MAX;
                itCount = 0;
            }
        }
        lastResult = result;
        //printf("thread:%d, result: %lf\n", omp_get_thread_num(), result);
    }
    double end = omp_get_wtime();

    printf("time: %lf\n", end - start);
    //printVector(nextX);

    free(A);
    free(b);
    free(prevX);
    free(Ax);
    free(nextX);
    return 0;
}