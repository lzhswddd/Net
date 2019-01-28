#include <iostream>
#include <iomanip>
#include "Mat.h"
using namespace std;
using namespace nn;


Matrix::Matrix()
{
	init();
	checkSquare();
}
Matrix::Matrix(int n)
{
	init();
	*this = Matrix(n, n, 1);
}
Matrix::Matrix(int row, int col)
{
	init();
	check(row, col);
	depth = 1;
	this->row = row;
	this->col = col;
	matrix = new double[row*col*depth];
	if (matrix != nullptr)
		memset(matrix, 0, length() * sizeof(double));
	checkSquare();
}
Matrix::Matrix(int row, int col, int depth)
{
	init();
	check(row, col);
	this->row = row;
	this->col = col;
	this->depth = depth;
	matrix = new double[row*col*depth];
	if (matrix != nullptr)
		memset(matrix, 0, length() * sizeof(double));
	checkSquare();
}
Matrix::Matrix(Size size_)
{
	init();
	check(size_.hei, size_.wid);
	row = size_.hei;
	col = size_.wid;
	depth = 1;
	matrix = new double[row*col*depth];
	if (matrix != nullptr)
		memset(matrix, 0, length() * sizeof(double));
	checkSquare();
}
Matrix::Matrix(Size3 size_)
{
	init();
	check(size_.x, size_.y, size_.z);
	row = size_.x;
	col = size_.y;
	depth = size_.z;
	matrix = new double[row*col*depth];
	if (matrix != nullptr)
		memset(matrix, 0, length() * sizeof(double));
	checkSquare();
}
Matrix::Matrix(double *matrix, int n)
{
	init();
	*this = Matrix(matrix, n, n);
}
Matrix::Matrix(int *matrix, int n)
{
	init();
	*this = Matrix(matrix, n, n);
}
Matrix::Matrix(int *matrix, int row, int col, int channel)
{
	init();
	if (matrix != nullptr) {
		check(row, col, channel);
		depth = 1;
		this->row = row;
		this->col = col;
		this->depth = channel;
		this->matrix = new double[row*col*depth];
		if (this->matrix != nullptr)
			for (int index = 0; index < length(); index++)
				this->matrix[index] = (double)matrix[index];

	}
	checkSquare();
}
Matrix::Matrix(double *matrix, int row, int col, int channel)
{
	init();
	if (matrix != nullptr) {
		check(row, col);
		depth = 1;
		this->row = row;
		this->col = col;
		this->depth = channel;
		this->matrix = new double[row*col*depth];
		if (this->matrix != nullptr)
			memcpy(this->matrix, matrix, row*col*depth * sizeof(double));

	}
	checkSquare();
}
Matrix::Matrix(const Matrix &src)
{
	init();
	setvalue(src);
	checkSquare();
}
Matrix::Matrix(const Matrix *src)
{
	init();
	if (src != nullptr)
		setvalue(*src);
	checkSquare();
}
Matrix::Matrix(Matrix a, Matrix b, X_Y_Z merge)
{
	init();
	if (merge == ROW) {
		if (a.col == b.col) {
			row = a.row + b.row;
			col = a.col;
			depth = a.depth;
			matrix = new double[row*col*depth];
			if (matrix != nullptr) {
				memcpy(matrix, a.matrix,
					a.row*a.col*a.depth * sizeof(double));
				memcpy(matrix + (a.row*a.col*a.depth), b.matrix,
					b.row*b.col*b.depth * sizeof(double));
			}
			else {
				matrix = nullptr;
			}			
		}
	}
	else if (merge == COL) {
		if (a.row == b.row) {
			row = a.row;
			col = a.col + b.col;
			depth = a.depth;
			matrix = new double[row*col*depth];
			double *temp = a.matrix;
			if (matrix != nullptr)
				for (int i = 0; i < row; i++) {
					memcpy(matrix + i * col*depth,
						a.matrix + i * a.col*depth,
						a.col*depth * sizeof(double));
					memcpy(matrix + i * col*depth + a.col*depth,
						b.matrix + i * b.col*depth,
						b.col*depth * sizeof(double));
				}
			else {
				matrix = nullptr;
			}
		}
	}
	checkSquare();
}
Matrix::Matrix(MatCommaInitializer_ & m)
{
	init();
	*this = Matrix(m.matrix(), m.rows(), m.cols(), m.channels());
}
Matrix::~Matrix()
{
	if (matrix != nullptr) {
		delete[]matrix;
		matrix = nullptr;
	}
}

double* Matrix::mat_()const
{
	return matrix;
}

void Matrix::DimCheck()const
{
	if (depth != 1) {
		cerr << errinfo[ERR_INFO_DIM] << endl;
		throw errinfo[0];
	}
}

void Matrix::copy(Matrix &src, int Row_Start, int Row_End, int Col_Start, int Col_End)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int hei = Row_End - Row_Start + 1;
	int wid = Col_End - Col_Start + 1;
	check(hei, wid);
	if (src.matrix == nullptr) {
		src = Mat(hei, wid, depth);
	}
	for (int y = Row_Start, j = 0; y <= Row_End; y++, j++)
		for (int x = Col_Start, i = 0; x <= Col_End; x++, i++)
			for (int z = 0; z < depth; z++)
				src(y, x, z) = matrix[(i + j * wid)*depth + z];
}

void Matrix::swap(Matrix &src)const
{
	src.setvalue(*this);
}

void Matrix::addones(direction dire)
{
	Matrix temp(row, col + 1, depth);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col + 1; j++) {
			for (int z = 0; z < depth; z++) {
				if (dire == LEFT) {
					if (j == 0)
						temp(i, j, z) = 1;
					else
						temp(i, j, z) = matrix[(j - 1 + i * col)*depth + z];
				}
				else if (dire == RIGHT) {
					if (j == col)
						temp(i, j, z) = 1;
					else
						temp(i, j, z) = matrix[(j - 1 + i * col)*depth + z];
				}
			}
		}
	}
	*this = temp;
}
void Matrix::mChannel(const Matrix & src, int channel)
{
	if (matrix == nullptr || src.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row != src.row || col != src.col || channel >= depth) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j) {
			matrix[(i*col + j)*depth + channel] = src(i, j);
		}
	}
}
void Matrix::reshape(int row, int col, int channel)
{
	if (matrix == nullptr) {
		fprintf(stderr, errinfo[ERR_INFO_EMPTY]);
		throw errinfo[ERR_INFO_EMPTY];
	}
	if (length() != row*col*channel) {
		fprintf(stderr, errinfo[ERR_INFO_UNLESS]);
		throw errinfo[ERR_INFO_UNLESS];
	}
	else {
		this->row = row;
		this->col = col;
		this->depth = channel;
	}
}
bool Matrix::setSize(int row, int col)
{
	if (length() == 0 && row*col > 0) {
		*this = Matrix(row, col, depth);
		return true;
	}
	if (length() == row * col) {
		this->row = row;
		this->col = col;
		return true;
	}
	return false;
}
void Matrix::setNum(double number, int index)
{
	checkindex(index);
	matrix[index] = number;
}
void Matrix::setNum(double number, int index_y, int index_x)
{
	checkindex(index_x, index_y);
	matrix[index_y*col + index_x] = number;
}
void Matrix::setMat(double *mat, int hei, int wid)
{
	if ((row <= 0 || col <= 0)) return;
	Matrix::~Matrix();
	*this = Matrix(mat, hei, wid);
	checkSquare();
}
void Matrix::setvalue(const Matrix &src)
{
	row = src.row;
	col = src.col;
	depth = src.depth;
	square = src.square;
	if (src.matrix != nullptr) {
		if (matrix != nullptr) {
			delete[]matrix;
			matrix = nullptr;
		}
		matrix = new double[row*col*depth];
		if (matrix != nullptr)
			memcpy(matrix, src.matrix, row*col*depth * sizeof(double));
	}
	else matrix = nullptr;
}
void Matrix::setOpp()
{
	*this = mOpp(*this);
}
void Matrix::setAdj()
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (isEnable() == 0)
		*this = Adj();
	else {
		cerr << errinfo[ERR_INFO_ADJ] << endl;
		throw errinfo[0];
	}
}
void Matrix::setTran()
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	*this = Tran();
}
void Matrix::setInv()
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (isEnable() == 0)
		*this = Inv();
	else {
		cerr << errinfo[ERR_INFO_INV] << endl;
		throw errinfo[0];
	}
}
void Matrix::setPow(int num)
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (isEnable() == 0)
		*this = Pow(num);
	else {
		cerr << errinfo[ERR_INFO_POW] << endl;
		throw errinfo[0];
	}
}
void Matrix::setIden()
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (isEnable() == 0)
		*this = eye(row);
	else {
		cerr << errinfo[ERR_INFO_IND] << endl;
		throw errinfo[0];
	}
}
Size3 Matrix::size3() const
{
	return Size3(row, col, depth);
}
int Matrix::rows()const
{
	return row;
}
int Matrix::cols()const
{
	return col;
}
int Matrix::channels() const
{
	return depth;
}
int Matrix::size()const
{
	return length();
}
Size Matrix::mSize()const
{
	return Size(row, col);
}
int Matrix::length()const
{
	return row * col*depth;
}
int Matrix::isEnable()const
{
	if (matrix == nullptr)
		return -1;
	if (!square)
		return -2;
	return 0;
}
bool Matrix::empty()const
{
	if (matrix == nullptr)return true;
	else return false;
}
bool Matrix::Square()const
{
	return square;
}
double& Matrix::at(int index_y, int index_x)const
{
	checkindex(index_x, index_y);
	return matrix[index_y*col + index_x];
}
double& Matrix::at(int index)const
{
	checkindex(index);
	return matrix[index];
}
int Matrix::toX(int index)const
{
	return index%col;
}
int Matrix::toY(int index)const
{
	return index/col;
}
double Matrix::frist()const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	return matrix[0];
}
double& Matrix::findAt(double value)const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	static double err = NAN;
	for (int ind = 0; ind < length(); ind++)
		if (matrix[ind] == value)
			return matrix[ind];
	return err;
}
double& Matrix::findmax()const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int max_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if (matrix[max_adr] < matrix[ind])
			max_adr = ind;
	return matrix[max_adr];
}
double& Matrix::findmin()const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int min_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if (matrix[min_adr] > matrix[ind])
			min_adr = ind;
	return matrix[min_adr];
}
int Matrix::find(double value)const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	for (int ind = 0; ind < length(); ind++)
		if (matrix[ind] == value)
			return ind;
	return -1;
}
int Matrix::maxAt()const
{
	if (empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int max_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if (matrix[max_adr] < matrix[ind])
			max_adr = ind;
	return max_adr;
}
int Matrix::minAt()const
{
	if (empty()) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	int min_adr = 0;
	for (int ind = 1; ind < length(); ind++)
		if (matrix[min_adr] > matrix[ind])
			min_adr = ind;
	return min_adr;
}
bool Matrix::contains(double value)const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	for (int ind = 0; ind < length(); ind++)
		if (matrix[ind] == value)
			return true;
	return false;
}

void Matrix::show()const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	cout.setf(ios::scientific);
	cout.setf(ios::showpos);
	cout.setf(ios::left);
	for (int z = 0; z < depth; z++) {
		for (int i = 0; i < row; i++) {
			cout << "[ ";
			for (int j = 0; j < col - 1; j++) {
				cout << setw(8) << setprecision(2) << setfill(' ') << matrix[(i*col + j)*depth + z] << ", ";
			}
			cout << setw(8) << setprecision(2) << setfill(' ') << matrix[(i*col + col - 1)*depth + z];
			cout << " ]" << endl;
		}
	}
	cout.unsetf(ios::scientific);
	cout.unsetf(ios::showpos);
	cout.unsetf(ios::left);
	cout << defaultfloat << setprecision(6);
}

const Matrix Matrix::Abs()const
{
	Matrix mat(this);
	return mAbs(mat);
}
const Matrix Matrix::Pow(int num)const
{
	Matrix temp(this);
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		temp = POW(temp, num);
	}
	else if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_POW] << endl;
		throw errinfo[0];
	}
	return temp;
}
double Matrix::Sum(int num, bool _abs)const
{
	if (isEnable() == -1) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	double sum = 0;
	if (num == 1) {
		for (int ind = 0; ind < length(); ind++)
			if (_abs)
				sum += fabs(matrix[ind]);
			else
				sum += matrix[ind];
	}
	else
		for (int ind = 0; ind < length(); ind++)
			if (_abs)
				sum += pow(fabs(matrix[ind]), num);
			else
				sum += pow(matrix[ind], num);
	return sum;
}
const Matrix Matrix::Opp()const
{
	Matrix mat(this);
	return mOpp(mat);
}
const Matrix Matrix::Adj()const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		Matrix mat(this);
		return adj(mat);
	}
	else {
		cerr << errinfo[ERR_INFO_ADJ] << endl;
		throw errinfo[0];
	}
}
const Matrix Matrix::Tran()const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mat(this);
	return tran(mat);
}
const Matrix Matrix::Inv()const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		Matrix mat(this);
		return inv(mat);
	}
	else {
		Matrix m(matrix, row, col);
		try {
			return pinv(m, RIGHT);
		}
		catch (...) {
			try {
				return pinv(m, LEFT);
			}
			catch (...) {
				cerr << errinfo[ERR_INFO_PINV] << endl;
				throw errinfo[0];
			}
		}
	}
}
const Matrix Matrix::Reverse()const
{
	Matrix mat(this);
	return reverse(mat);
}
double Matrix::Det()
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square)
		return det(*this);
	else
		return NAN;
}
double Matrix::Norm(int num)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (col != 1 && row != 1) {
		cerr << errinfo[ERR_INFO_NORM] << endl;
		throw errinfo[0];
	}
	if (num < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (num == 1)
		return Sum(1, true);
	else if (num == 2)
		return sqrt(Sum(2, true));
	//else if (isinf(num) == 1)
	//	return abs(matrix[find(findmax())]);
	//else if (isinf(num) == -1)
	//	return abs(matrix[find(findmin())]);
	else
		return pow(Sum(num, true), 1 / double(num));
}
double Matrix::Matrix::Cof(int x, int y)
{
	return cof(*this, x, y);
}
double Matrix::EigenvalueMax(double offset)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		int count = 0;
		double err = 100 * offset;
		Matrix v;
		Matrix u0 = ones(row, 1);
		while (err > offset) {
			v = *this*u0;
			Matrix u1 = v * (1 / v.findmax());
			err = (u1 - u0).Abs().findmax();
			u0 = u1; count += 1;
			if (count >= 1e+3) {
				cerr << errinfo[ERR_INFO_EIGEN] << endl;
				throw errinfo[0];
			}
		}
		return v.findmax();
	}
	else {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw errinfo[0];
	}
}
double Matrix::RandSample()
{
	return mRandSample(*this);
}
const Matrix Matrix::EigenvectorsMax(double offset)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (square) {
		int count = 0;
		double err = 100 * offset;
		Matrix v;
		Matrix u0 = ones(row, 1);
		while (err > offset) {
			v = *this*u0;
			Matrix u1 = v * (1 / v.findmax());
			err = (u1 - u0).Abs().findmax();
			u0 = u1; count += 1;
			if (count >= 1e+3) {
				cerr << errinfo[ERR_INFO_EIGEN] << endl;
				throw errinfo[0];
			}
		}
		return u0;
	}
	else {
		cerr << errinfo[ERR_INFO_SQUARE] << endl;
		throw errinfo[0];
	}
}
const Matrix Matrix::Exp()const
{
	Matrix mat(this);
	return mExp(mat);
}

const Matrix Matrix::Log()const
{
	Matrix mat(this);
	return mLog(mat);
}

const Matrix Matrix::Sqrt()const
{
	Matrix mat(this);
	return mSqrt(mat);
}

void Matrix::init()
{
	row = 0;
	col = 0;
	depth = 0;
	matrix = nullptr;
}
void Matrix::checkSquare()
{
	if (row == col) 
		square = true;
	else 
		square = false;
}
void Matrix::checkindex(int index)const
{
	if (row == 0 || col == 0) {
		cerr << errinfo[ERR_INFO_LEN] << endl;
		throw errinfo[0];
	}
	if (index > length() - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (index < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
}
void Matrix::checkindex(int index_x, int index_y)const
{
	if (row == 0 || col == 0) {
		cerr << errinfo[ERR_INFO_LEN] << endl;
		throw errinfo[0];
	}
	if (index_x < 0 || index_y < 0) {
		cerr << errinfo[ERR_INFO_UNLESS] << endl;
		throw errinfo[0];
	}
	if (index_x*col + index_y > row*col - 1 || index_x >= row || index_y >= col) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
}

const Matrix Matrix::operator + (const double val)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, depth);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < depth; z++)
				mark((j + i * col)*depth + z) = matrix[(j + i * col)*depth + z] + val;
	return mark;
}
const Matrix Matrix::operator + (const Matrix &a)const
{
	if (matrix == nullptr || a.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row == 1 && col == 1 && depth == 1) {
		return (*this)(0) + a;
	}
	else if (a.row == 1 && a.col == 1 && a.depth == 1) {
		return *this + a(0);
	}
	else if (a.row == 1 && a.col == 1 && a.depth == depth) {
		Matrix mat(row, col, depth);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				for (int z = 0; z < depth; z++)
					mat((j + i * col)*depth + z) = matrix[(j + i * col)*depth + z] + a(z);
		return mat;
	}
	if (row != a.row || col != a.col || depth != a.depth) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, depth);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < depth; z++)
				mark((j + i * col)*depth + z) = matrix[(j + i * col)*depth + z] + a((j + i * col)*depth + z);
	return mark;
}
void Matrix::operator += (const double val)
{
	*this = *this + val;
}
void Matrix::operator+=(const Matrix & a)
{
	*this = *this + a;
}
const Matrix Matrix::operator-(void) const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(this);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < depth; z++)
				mark((j + i * col)*depth + z) = -mark((j + i * col)*depth + z);
	return mark;
}
const Matrix Matrix::operator - (const double val)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, depth);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < depth; z++)
				mark((j + i * col)*depth + z) = matrix[(j + i * col)*depth + z] - val;
	return mark;
}
const Matrix Matrix::operator - (const Matrix &a)const
{
	if (matrix == nullptr || a.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row == 1 && col == 1 && depth == 1) {
		return (*this)(0) - a;
	}
	else if (a.row == 1 && a.col == 1 && a.depth == 1) {
		return *this - a(0);
	}
	else if (a.row == 1 && a.col == 1 && a.depth == depth) {
		Matrix mat(row, col, depth);
		for (int i = 0; i < row; i++)
			for (int j = 0; j < col; j++)
				for (int z = 0; z < depth; z++)
					mat((j + i * col)*depth + z) = matrix[(j + i * col)*depth + z] - a(z);
		return mat;
	}
	if (row != a.row || col != a.col || depth != a.depth) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, depth);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < depth; z++)
				mark((j + i * col)*depth + z) = matrix[(j + i * col)*depth + z] - a((j + i * col)*depth + z);
	return mark;
}
void Matrix::operator-=(const double val)
{
	*this = *this - val;
}
void Matrix::operator-=(const Matrix & a)
{
	*this = *this - a;
}
const Matrix Matrix::operator * (const double val)const
{
	if (matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, depth);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < depth; z++)
				mark((j + i * col)*depth + z) = matrix[(j + i * col)*depth + z] * val;
	return mark;
}
const Matrix Matrix::operator * (const Matrix &a)const
{
	if (matrix == nullptr || a.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row == 1 && col == 1 && depth == 1) {
		return (*this)(0) * a;
	}
	else if (a.row == 1 && a.col == 1 && a.depth == 1) {
		return *this * a(0);
	}
	if (col != a.row) {
		cerr << errinfo[ERR_INFO_MULT] << endl;
		throw errinfo[0];
	}
	if (depth != a.depth) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, a.col, depth);
	for (int z = 0; z < depth; z++)
		for (int i = 0; i < row; i++)
			for (int j = 0; j < a.col; j++) {
				double temp = 0;
				for (int d = 0; d < col; d++)
					temp = temp + matrix[(i*col + d)*depth + z] * a((j + d * a.col)*depth + z);
				mark((j + i * a.col)*depth + z) = temp;
			}
	return mark;
}
void Matrix::operator*=(const double val)
{
	*this = *this * val;
}
void Matrix::operator*=(const Matrix & a)
{
	*this = *this * a;
}
const Matrix Matrix::operator / (const double val)const
{
	return (*this) * (1.0 / val);
}
const Matrix Matrix::operator / (const Matrix &a)const
{
	if (matrix == nullptr || a.matrix == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	if (row == 1 && col == 1 && depth == 1) {
		return (*this)(0) / a;
	}
	else if (a.row == 1 && a.col == 1 && a.depth == 1) {
		return *this / a(0);
	}
	if (row != a.row || col != a.col || depth != a.depth) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw errinfo[0];
	}
	Matrix mark(row, col, depth);
	for (int i = 0; i < row; i++)
		for (int j = 0; j < col; j++)
			for (int z = 0; z < depth; z++)
				mark((j + i * col)*depth + z) = matrix[(j + i * col)*depth + z] / a((j + i * col)*depth + z);
	return mark;
}
void Matrix::operator/=(const double val)
{
	*this = *this / val;
}
void Matrix::operator/=(const Matrix & a)
{
	*this = *this / a;
}
void Matrix::operator = (const Matrix &temp)
{
	setvalue(temp);
}
bool Matrix::operator == (const Matrix &a)const
{
	if (col != a.col) {
		return false;
	}
	if (row != a.row) {
		return false;
	}
	if (depth != a.depth) {
		return false;
	}
	if (memcmp(matrix, a.matrix, col*row*depth * sizeof(double)) == 0)
		return true;
	return false;
}
bool Matrix::operator != (const Matrix & a)const
{
	return !(*this == a);
}
double & Matrix::operator()(const int index) const
{
	if (index >length() - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (index < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	return matrix[index];
}
double& Matrix::operator()(const int row, const int col) const
{
	if (row > this->row - 1|| col > this->col - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (row < 0|| col < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	return matrix[(row*this->col + col)*depth];
}
double & Matrix::operator()(const int row, const int col, const int depth) const
{
	if (row > this->row - 1 || col > this->col - 1 || depth > this->depth - 1) {
		cerr << errinfo[ERR_INFO_MEMOUT] << endl;
		throw errinfo[0];
	}
	if (row < 0 || col < 0 || depth < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	return matrix[(row*this->col + col)*this->depth + depth];
}
const Matrix Matrix::operator()(const int index, X_Y_Z rc) const
{
	if (index < 0) {
		cerr << errinfo[ERR_INFO_VALUE] << endl;
		throw errinfo[0];
	}
	if (!matrix) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw errinfo[0];
	}
	switch (rc) {
	case ROW:
		if (index > this->row - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw errinfo[0];
		}
		return Block(*this, index, index, 0, col - 1);
	case COL:
		if (index > this->col - 1) {
			cerr << errinfo[ERR_INFO_MEMOUT] << endl;
			throw errinfo[0];
		}
		return Block(*this, 0, row - 1, index, index);
	default:return Matrix();
	}
}
const Matrix Matrix::operator [] (const int channel)const
{
	return mSplit(*this, channel);
}



