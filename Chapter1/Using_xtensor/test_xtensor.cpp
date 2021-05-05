#include <iostream>
#include <xtensor/xarray.hpp> // init xarray
#include <xtensor/xio.hpp>  // print out matrix object
#include <xtensor/xlayout.hpp> // xt::layout_type
#include <xtensor/xadapt.hpp> // xt::adapt for mat.shape()
#include <xtensor/xview.hpp> // xt::view for indexing, slicing
#include <xtensor/xslice.hpp> // for placeholder ~ xt::placeholder
#include <xtensor/xrandom.hpp> // xt::random::rand() 
#include <xtensor-blas/xlinalg.hpp>  // xt::linalg::dot for dot operation
#include <xtensor/xmanipulation.hpp> // for xt::transpose matrix
#include <xtensor/xtensor_forward.hpp> // xt::xtensor_fixed
#include <vector>

using namespace std;
using namespace xt::placeholders; // using xt::placeholder::_ 

int main()
{
    // declaration
    {
        cout << "==========DECLARATION===========\n" << endl;
        vector<size_t> shape = { 3, 2, 4 };
		xt::xarray<double, xt::layout_type::row_major> a(shape);
		cout << a << endl;

		// initialize matrix 3x4 fill with zeros
		xt::xarray<double> mat = xt::zeros<double>({ 3,4 });
		cout << "\n\n";
		cout << mat << endl;

		// initialize matrix 1x5 fill with ones
		xt::xarray<int> mat_x = xt::ones<double>({ 1, 5 });
		cout << "\n\n";
		cout << mat_x << endl;

		xt::xarray<int> mat_y = xt::linspace<int>(1, 10, 100);
		cout << "\n\n";
		cout << mat_y << endl;

		xt::xarray<double> mat_1 = xt::arange(0, 10);
		cout << "\n\n";
		cout << mat_1 << endl;
    }
    // container
    {
        cout << "\n\n==========CONTAINER===========\n" << endl;
        xt::xarray<double> mat_2 = {	{ 3, 4, 5, 6 }, 
										{ 5, 6, 6, 9 },
										{ 1, 3, 5, 0 } };
		cout << "\nMat_2: \n" << mat_2 << endl;

		// display shape [2, 3]
		auto shape = mat_2.shape();
		cout << "\nShape: " << xt::adapt(shape) << endl;

		// dimension
		//auto dim = mat_2.dimension();
		cout << "\nDimension: " << mat_2.dimension() << endl;
		
		// size
		cout << "\nSize: " << mat_2.size() << endl;

		// index in shape
		cout << "\nShape(0) and Shape(1): " << mat_2.shape(0) << " " << mat_2.shape(1) << endl;

		// reshape
		mat_2.reshape({ 2, 6 });
		cout << "\nReshape mat_2 2x6: \n" << mat_2 << endl;

		mat_2.reshape({ 4, 3 });
		cout << "\nReshape mat_2 4x3: \n" << mat_2 << endl;

		xt::xarray<double> mat_3 = { {1, 2}, {3, 5}, {6, 7} };
		mat_3.reshape({ 2, -1 });
		cout << "\nNew mat_3: \n" << mat_3 << endl;

		// resize
		xt::xarray<double> mat_4 = { 1, 2, 3, 5 };
		cout << "\nmat_4: \n" << mat_4 << endl;
		mat_4.resize({ 2, 3 });
		cout << "\nResized mat_4: \n" << mat_4 << endl;

		// fill
		auto mat_5 = xt::xarray<double>::from_shape({ 2, 3 });
		mat_5.fill(6);
		cout << "\nFilled all elements in mat_5 with 6: \n" << mat_5 << endl;
    }
    // slicing and indexing
    {
        cout << "\n\n==========SLICING AND INDEXING===========\n" << endl;
        xt::xarray<int> mat_6 = { {4, 5, 6}, {7,8,9} };

		// mat_6[1,2] = 9
		cout << "\nmat_6[1,2]: " << mat_6(1, 2) << endl;

		// mat_6[1] = {7,8,9} ( xt::view(mat_6, 3, xt::all()) )
		cout << "\nmat_6[1]: " << xt::row(mat_6, 1) << endl;

		// mat_6[:, 1] = {5, 8} ( xt::view(mat_6, xt::all(), 1 )
		cout << "\nmat_6[:,1]: " << xt::col(mat_6, 1) << endl;

		xt::xarray<int> mat_7 = xt::arange(0, 18);
		cout << "\nmat_7: \n" << mat_7 << endl;
		mat_7.reshape({ 3, 6 });
		cout << "\nmat_7 3x6: \n" << mat_7 << endl;
		// mat_7[:2, 3:]
		cout << "\nmat_7[:2, 3:]: \n" << xt::view(mat_7, xt::range(0, 2), xt::range(3, _)) << endl;
    }
    // arithmetic operations
    {
        cout << "\n\n==========ARITHMETIC OPERATIONS===========\n" << endl;

        xt::xarray<double> new_mat1 = xt::random::rand<double>({ 2,3 });
		xt::xarray<double> new_mat2 = xt::random::rand<double>({ 2,3 });
		cout << "\nRandom mat1 2x3: \n" << new_mat1 << endl;
		cout << "\nRandom mat2 2x3: \n" << new_mat2 << endl;

		xt::xarray<double> kq = new_mat1 + new_mat2;
		cout << "\nSum of new_mat1 and new_mat2: \n" << kq << endl;

		// new_mat1 = new_mat1 - new_mat2
		new_mat1 -= new_mat2;
		cout << "\nnew_mat1 -= new_mat2: \n" << new_mat1 << endl;

		// matrix multiplication
		kq = new_mat1 * new_mat2;
		cout << "\nmew_mat1 * new_mat2: \n" << kq << endl;

		// dot operation
		// both shape of new_mat1 and new_mat2 are {2,3}
		// transpose new_mat2 to shape {3,2} so that {2,3} * {3,2} = {2,2} matrix
		new_mat2 = xt::transpose(new_mat2);
		auto shape_mat1 = new_mat1.shape();
		auto shape_mat2 = new_mat2.shape();
		cout << "\nShape new_mat1: " << xt::adapt(shape_mat1) << ", shape new_mat2: " << xt::adapt(shape_mat2) << endl;
        // dot operation
		kq = xt::linalg::dot(new_mat1, new_mat2);
		cout << "\nnet_mat1 dot new_mat2: \n" << kq << endl;

		kq = new_mat2 + 5;
		cout << "\nnew_mat2 + 5: \n" << kq << endl;
    }
	// xtensor is multidimensional array whose dims are fixed at
	// compilation time.
	{
		array<size_t, 3> shape1 = { 3, 2, 4 };
		xt::xtensor<double, 3, xt::layout_type::row_major > ten_1 = xt::random::rand<double>(shape1);
		cout << "\ntensor shape 3, 2, 4 with random numbers init: \n" << ten_1 << endl;

		cout << "\nCheck ten_1 shape: " << xt::adapt(ten_1.shape()) << endl;
		cout << "\nCheck ten_1 dimension: " << ten_1.dimension() << endl;

		// xtensor_fixed type is multidim array with dim shape fixed
		// at compile time
		xt::xtensor_fixed<double, xt::xshape<3, 2, 4>> ten_2;
		cout << "\ntensor_fixed shape 3, 2, 4: \n" << ten_2 << endl;
		auto ten_2_shape = ten_2.shape();
	}
    return 0;
};