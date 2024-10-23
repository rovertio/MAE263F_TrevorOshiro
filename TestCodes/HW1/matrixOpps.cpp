#include "matrixOpps.h"
// #include<gsl>

#include <iostream>
#include <vector>
#include <cmath>

typedef std::vector<std::vector<float> > Mat;
using namespace std; 
// template <std::size_t N>


Mat mat_Mult(Mat (*mat1), int mat1R, int mat1C, Mat (*mat2), int mat2R, int mat2C){

    return *mat1;
}

// Prints out matrices
void mat_Print(Mat *mat, int row, int col){
    for (int i = 1; i <= row; i++){
        for (int j = 1; j <= col; j++){
            std::cout << (*mat)[i-1][j-1] << " ";
        }
        std::cout << "\n";
    }
}

