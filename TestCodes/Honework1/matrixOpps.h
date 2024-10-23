#ifndef MATRIXOPPS_H
#define MATRIXOPPS_H

#include <iostream>
 // using namespace std; 
template <std::size_t N>


float matMult(float (&mat1)[N], int mat1R, int mat1C, float (&mat2)[N], int mat2R, int mat2C);

#endif