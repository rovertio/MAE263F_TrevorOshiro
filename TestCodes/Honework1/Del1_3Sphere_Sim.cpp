#include "matrixOpps.h"
#include <iostream>
template <std::size_t N>


void main() {
    using namespace std; 

    // float a[2][3] = {1.0, 2, 2, 3, 3, 4};
    // float b[2][3] = {1.0, 2, 2, 3, 3, 4};

    float a[3] = {1.0, 2, 2};
    float b[3] = {1.0, 2, 2};

    cout << &a;
    // matMult(a, 2, 3, b, 2, 3);
}