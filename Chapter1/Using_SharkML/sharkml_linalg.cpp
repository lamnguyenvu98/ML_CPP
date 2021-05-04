#include <shark/LinAlg/BLAS/remora.hpp>
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    // definition
    {
        cout << "=========== DEFINITION ==========\n" << endl;

        // dynamic sized array dense vector
        remora::vector<double> a1(100, 1.0); // vector size 100 and filled with 1.0
        cout << "\nvector a1: \n" << a1 << endl;
    }
    return 0;
}