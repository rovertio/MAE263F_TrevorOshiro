#ifndef MATRIXOPPS_H
#define MATRIXOPPS_H

#include <iostream>
#include <vector>
#include <cmath>

 // using namespace std; 
typedef std::vector<std::vector<float> > Mat;
using namespace std; 
template <std::size_t N>


Mat mat_Mult(Mat (*mat1), int mat1R, int mat1C, Mat (*mat2), int mat2R, int mat2C);


void mat_Print(Mat *mat, int row, int col);



#endif