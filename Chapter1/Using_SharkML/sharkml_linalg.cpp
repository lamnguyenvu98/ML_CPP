#include <shark/LinAlg/BLAS/remora.hpp>
#include <iostream>
#include <vector>

using namespace std;

int main()
{
    // definition
    {
        cout << "=========== DEFINITION ==========" << endl;

        // dynamic sized array dense vector
        remora::vector<double> a1(100, 1.0); // vector size 100 and filled with 1.0
        cout << "\nvector a1: \n" << a1 << endl;

        // dynamically sized array dense matrix
        remora::matrix<double> a2(2, 2); // 2x2 matrix
        cout << "\nMatrix a2: \n" << a2 << endl;
    }
    // initialization
    {
        cout << "\n\n=========== INITILIZATION ==========" << endl;

        // fill with one value
        remora::matrix<float> b1(2,2, 2.5f);
        cout << "\nb1 matrix 2x2 filled with 2.5: \n" << b1 << endl;

        // initializer list
        remora::matrix<float> b2{{1, 1}, {1, 1}}; // 2x2 matrix ones
        cout << "\nb2 matrix 2x2 filled with 1: \n" << b2 << endl;
        remora::matrix<float> b3{{1, 2}, {3, 4}};
        cout << "\nb3 matrix initializer list: \n" << b3 << endl;

        // wrap c++ array
        vector<float> data{1, 2, 3, 4};
        auto b4 = remora::dense_matrix_adaptor<float>(data.data(), 2, 2);
        std::cout << "\nWrapped array matrix b4: \n" << b4 << std::endl;
        auto b5 = remora::dense_vector_adaptor<float>(data.data(), 4);
        std::cout << "\nWrapped array vector b5 \n" << b5 << std::endl;

        cout << "\n\n=========== ACCESSING ELEMENTS ==========" << endl;

        b4(0,0) = 3.14f;
        cout << "\nMatrix b4 after change element b4(0,0): \n" << b4 << endl;
        cout << "\nb5: \n" << b5 << endl;
        b5(1) = 3.33f;
        cout << "\nVector b5 after change element b5(1): \n" << b5 << endl;
        cout << "\nb4: \n" << b4 << endl;
        cout << "\n=======> Weird. Modify b5 also modify element in b4 " << endl;
        // Weird. Modify b5 also modify element in b4 :/
    }
    // arithmetic operations
    {
        cout << "\n\n=========== ARITHMETIC OPERATIONS ==========" << endl;

        remora::matrix<float> c1{ {1, 1}, {1, 1} };
        remora::matrix<float> c2{ {2, 2}, {2, 2} };
        remora::matrix<float> c3 = c1 + c2;
        cout << "\nc3 = c1 + c2 \n" << c3 << endl;

        c1 -= c2;
        cout << "\nc1 -= c2 => c1 = \n" << c1 << endl;

        c1 = { {1, 1}, {1, 1} }; // reinitialize

        // dot product 2 ways
        c3 = remora::prod(c1, c2);
        cout << "\n1: c3 = c1 dot c2 \n" << c3 << endl;
        c3 = c1 % c2;
        cout << "\n2: c3 = c1 % c2 \n" << c3 << endl;
    }
    // partial access
    {
        cout << "\n\n=========== PARTIAL ACCESS ==========" << endl;

        remora::matrix<float> d1 = {{1, 2, 3, 4}, \
                                    {5, 6, 7, 8}, \
                                    {9, 10,  11, 12}, \
                                    {13, 14, 15, 16}}; // 4x4
        auto r = remora::rows(d1, 0, 2); // take row 0 -> 1 and form new matrix 2x4
        cout << "\nMatrix d1 rows 0,2: \n" << r << endl;

        auto sr = remora::subrange(d1, 1, 3, 1, 3); // (1->2, 1->2)
        cout << "\nMatrix subrange(1,3,1,3): \n" << sr << endl;
        sr *= 67;
        cout << "\nMatrix with updated subrange: \n" << d1 << endl;
    }
    // broadcasting is not supported directly
    {
        cout << "\n\n=========== BROADCASTING ==========" << endl;

        // reductions
        remora::matrix<float> e1{{1, 2, 3, 4}, {5, 6, 7, 8}}; //2x4
        remora::vector<float> e2{10, 10};
        auto cols = remora::as_columns(e1);
        cout << "\nSum reduction for columns\n" << remora::sum(cols) << endl;

        // update matrix rows
        // e1 shape 2x4
        // e1.size2() = 4
        for (size_t i = 0; i < e1.size2(); ++i) 
        {
            // take each column of e1 sum with e2
            remora::column(e1, i) += e2;
        }
        cout << "\nUpdated rows\n" << e1 << endl;
    }
    return 0;
}