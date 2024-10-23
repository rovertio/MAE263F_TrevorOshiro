// #include "matrixOpps.h"
#include "matrixOpps.cpp"
#include <vector>
#include <iostream>
#include <cmath>

using namespace std; 
typedef std::vector<std::vector<float> > Mat;

void main() {
    

    // float a[2][3] = {1.0, 2, 2, 3, 3, 4};
    // float b[2][3] = {1.0, 2, 2, 3, 3, 4};

    /*
    float a[3] = {1.0, 2, 2};
    float b[3] = {1.0, 2, 2};
    float c = 1.2;
    */
    Mat a {{1,2},{3,4}};
    Mat b {{1,2},{1,2}};

    // cout << &a << "\n";
    // cout << *a;
    //matMult(&a, 2, 3, &b, 2, 3);
    //printarray(c);

    mat_Print(&a, 2, 2);

    
    // std::cout << a[0][0] << " ";
    // std::cout << a[0][1] << "\n";
    // std::cout << a[1][0] << " ";
    // std::cout << a[1][1] << "\n";
    // std::cout << &a;
    

}