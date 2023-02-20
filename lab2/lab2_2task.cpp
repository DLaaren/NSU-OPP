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
    for (unsigned int i = 0; i < N; i++) {
        resM[i] = 0;
        for (unsigned int j = 0; j < N; j++) {
            resM[i] += M1[i*N + j] * M2[j];
        }
    }
}

void scalMult(double *M, double scalar) {
    for (unsigned int i = 0; i < N; i++) {
        M[i] *= scalar;
    }
}

void sub(const double *M1, const double *M2, double *resM) {
    for (unsigned int i = 0; i < N; i++) {
        resM[i] = M1[i] - M2[i];
    }
}

double EuclideanNorm(const double *M) {
    double norm = 0;
    for (unsigned int i = 0; i < N; i++) {
        norm += M[i]*M[i];
    }
    return sqrt(norm);
}

void copyVector(const double *M1, double *M2) {
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

    double *prevX;
    double *Ax;
    double *nextX;

    double tau = PARAMETER;
    double result = INT_MAX;
    double lastResult = INT_MAX;
    unsigned int itCount = 0;
    double norm_b = EuclideanNorm(b); // ||b||
    double norm_Ax = 0;
    double start = omp_get_wtime();

    //omp_set_num_threads(2); //omg the number of threads effects on the time!!!

    #pragma omp parallel shared(A, b, norm_b) private(prevX, nextX, Ax, norm_Ax, itCount) firstprivate(lastResult, result)
    {
        prevX = (double*)malloc(sizeof(double) * N);
        generateMatrix(prevX, N, 1);
        nextX = (double*)malloc(sizeof(double) * N);
        copyVector(prevX, nextX);
        Ax = (double*)malloc(sizeof(double) * N);

        while (result > EPSILON) {
            copyVector(nextX, prevX);
            mul(A, prevX, Ax); // Ax = A * prevX; 
            sub(Ax, b, Ax); // Ax - b

            norm_Ax = EuclideanNorm(Ax); // ||Ax - b||

            scalMult(Ax, tau); // tau * (Ax - b)
            sub(prevX, Ax, nextX); // x - tau * (Ax - b)

            //#pragma omp critical
            //{ 
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
                
            //}
        }
        //printf("thread:%d, final result: %lf\n", omp_get_thread_num(), result);
        double end = omp_get_wtime();
        printf("time: %lf\n\n", end - start);

        free(prevX);
        free(Ax);
        free(nextX);
    }
    //double end = omp_get_wtime();
    //printf("thread:%d, final result: %lf\n", omp_get_thread_num(), result);
    //printf("time: %lf\n", end - start);
    //printVector(nextX);

    free(A);
    free(b);
    return 0;
}