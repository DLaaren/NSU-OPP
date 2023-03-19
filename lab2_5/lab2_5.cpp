#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <climits>
#include <stdlib.h>
#include <omp.h>

void printArray(double* array, long int arraySize) {
    for (long int i = 0; i < arraySize; i++) {
        std::cout << array[i] << " ";
    }
    std::cout << "\n";
}

void createArray(double* array, long int arraySize) {
    for (long int i = 0; i < arraySize; i++) {
        array[i] = ((double)rand() / RAND_MAX) * 100;
        int sign = rand() % 2;
        if (sign == 0) {
            array[i] = -(array[i]);
        }
    }
}

bool isSorted(double* array, long int arraySize) {
    for (long int i = 0; i < arraySize - 1; i++) {
        if (array[i] > array[i+1]) {
            return false;
        }
    }
    return true;
}

int partition(double* array, long int l_bound, long int r_bound){
    double pivot = array[l_bound];
    long int left = l_bound;
    long int right = r_bound;
    while (left < right){
        // increase until you get to a point where the element is greater that the pivot
        while (left < r_bound && array[left] <= pivot) {
            ++left;
        }
        // increase until you get to a point where the element is less or equal to the pivot
        while (right >= 0 && array[right] > pivot) {
            --right;
        }
        // swap if within bounds
        if (left < right && left < r_bound && right >= 0) {
            std::swap(array[left], array[right]);
        }
    }
    // swap at the end
    if (left < right && left < r_bound && right >= 0) {
        std::swap(array[left], array[right]);
    }
    array[l_bound] = array[right];
    array[right] = pivot;
    return right;
}

void quickSort(double* array, long int left, long int right, int taskSize) {
    if (left < right) {
        long int q = partition(array, left, right);
        #pragma omp task shared(array) if (q - left  > taskSize)
        quickSort(array, left, q - 1, taskSize);
        #pragma omp task shared(array) if (right - q > taskSize)
        quickSort(array, q + 1, right, taskSize);
    }
}

int main() {
    srand(12345);
    int taskSize;
    long int arraySize;

    std::cout << "ArraySize:\tTaskSize\tTime:(milliseconds)\n";

    for (arraySize = 1000; arraySize <= 100000000; arraySize *= 10) {
    for (taskSize = 10000; taskSize <= 10000; taskSize *= 10) {
        double *array = new double[arraySize];
        createArray(array, arraySize);

        double start = omp_get_wtime();
        #pragma omp parallel num_threads(8)
        {
            #pragma omp single
                quickSort(array, 0, arraySize, taskSize);
            #pragma omp taskwait
        }
        double end = omp_get_wtime();

        if (isSorted(array, arraySize) == true) {
            std::cout << arraySize << "\t\t" << taskSize << "\t\t" << (end - start) * 1000 << "\n";
        } else {
            std::cout << "Wrong answer!\n";
        }
        delete [] array; 
    } 
    std::cout << "\n";
    }
    return 0;
}