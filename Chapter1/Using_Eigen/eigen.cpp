#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <vector>

using namespace std;
using namespace Eigen;

typedef Matrix<float, 3, 3> MyMatrix33f;
typedef Matrix<float, 3, 1> MyVector3f; // 3 rows, 1 col
typedef Matrix<float, Dynamic, Dynamic> MyMatrixf;

int main()
{
	{
		// decalration

		MyMatrix33f matrix33f;
		MyVector3f vec3f;
		MyMatrixf matrixx(10, 15);

		matrix33f = MyMatrix33f::Zero(); // fill matrix elements with zeros
		cout << "matrix zeros: \n" << matrix33f << endl;

		matrix33f = MyMatrix33f::Identity(); // fill matrix as identity matrix
		cout << "\nmatrix identity: \n" << matrix33f << endl;

		vec3f = MyVector3f::Random(); // fill vector elements with random values
		cout << "\nrandom vec: \n" << vec3f << endl;

		// Initialize matrix
		matrix33f << 1, 2, 3,
			4, 5, 6,
			7, 8, 9;
		cout << "\nInitialize matrix 3x3: \n" << matrix33f << endl;

		// access elements
		matrix33f(0, 0) = matrix33f(0, 0) + matrix33f(0, 1);
		cout << "\nAccess matrix 3x3: \n" << matrix33f << endl;

		// Use Eigen::Map to wrap C++ array or vector in Matrix type object
		int data[] = { 1, 2, 3, 5 };
		Eigen::Map<RowVectorXi> v(data, 4);
		cout << "\narray transfrom to eigen object: \n" << v << endl;

		vector<float> vec{ 1,2,3,4,5,6,7,8,9 };
		Eigen::Map<MyMatrix33f> a1(vec.data());
		cout << "\nVector transform to matrix 3x3 eigen: \n"
			<< a1 << endl;

		Eigen::Map<MyMatrixf> b1(vec.data(), 3, 3);
		cout << "\nVector to Matrix dynamic: \n" << b1 << endl;
		cout << "\n" << b1.transpose() << endl;
	}
	// arithmetic
	{
		Matrix2d a;
		a << 1, 2, 3, 4;
		Matrix2d b;
		b << 1, 2, 3, 4;
		Matrix2d result = a + b;
		cout << "\na + b = \n" << result << endl;

		Matrix2d wise_multi = a.array() * b.array();
		cout << "\nElement wise multiplication: \n" << wise_multi << endl;

		Matrix2d wise_div = a.array() / b.array();
		cout << "\nElement wise divide: \n" << wise_div << endl;

		a += b;
		cout << "\nNew matrix a: \n" << a << "\n\n" << b << endl;

		result = a * b; // matrix multiplication
		cout << "\nMatrix multiplication: \n" << result << endl;

		a = b.array() * 4;
		cout << "\nNew matrix a: \n" << a << "\n\n" << b << endl;
	}
	// patial access
	{
		MatrixXf m = MatrixXf::Random(4,4);
		cout << "\nMatrix m: \n" << m << endl;

		MatrixXf m_copy = m.block(0, 1, 2, 2);
		cout << "\nCopied Matrix m: \n" << m_copy << endl;
		cout << "\nMatrix block at (0,1,2,1): \n" << m.block(0, 1, 2, 1) << endl;

		// change matrix block in matrix m
		m.block(1, 1, 2, 2) *= 0;
		cout << "\nNew matrix m: \n" << m << endl;

		// another way to access matrix
		m_copy.row(1).array() *= 0;
		cout << "\nNew copied matrix m: \n" << m_copy << endl;

		cout << "\nRow 0 m_copy: \n" << m_copy.row(0) << endl;
		cout << "\nCol 1 m_copy:\n" << m_copy.col(1) << endl;
	}
	// broadcasting
	{
		MatrixXf mat(2, 3);
		mat << 1, 2, 3, 4, 5, 6;
		cout << "\nMat: \n" << mat << endl;
		VectorXf v(2);
		v << 1, 1;
		RowVectorXf rv(3);
		rv << 2, 2, 0;

		mat.colwise() += v;
		cout << "\nSum broadcasted over columns: \n" << mat << endl;

		mat.rowwise() += rv;
		cout << "\nSum broadcasted over row: \n" << mat << endl;

	}
	return 0;
};