#include <iostream>
#include "matrix.h"
#include <iomanip>
#include <time.h>
#include <unistd.h>

void printMatrix(const Matrix& m){
    for(int i = 0; i < m.getRowCount(); ++i){
        for(int j = 0; j < m.getColCount(); ++j){
            if(m(i, j) < 0.00001 && m(i, j) > - 0.00001) std::cout << 0 << "\t";
            else std::cout << std::setprecision(3) << m(i, j) << '\t';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


int main()
{
    Matrix::init(8);


    timespec t0;
    timespec t1;
    //test();
    clock_gettime(1, &t0);
    Matrix m2(3, 3);
    m2.setRandom(-1, 8, true);
    printMatrix(m2);
    clock_gettime(1, &t0);
    printMatrix(m2.getInverse());
    printMatrix(m2 * m2.getInverse());
    clock_gettime(1, &t1);
    std::cout << t1.tv_sec + t1.tv_nsec/1000000000.0 - t0.tv_sec - t0.tv_nsec/1000000000.0<< std::endl;

    return 0;
}
