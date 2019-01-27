#include "Function.h"
#include "Train.h"
using nn::Mat;

const Mat nn::Softmax(const Mat &y)
{
	Mat y_ = y;
	y_ -= Max(y_);
	Mat y_exp = mExp(y_);
	double y_sum = y_exp.Sum();
	Mat out = y_exp / y_sum;
	return out;
}

const Mat nn::Quadratic(const Mat &y, const Mat &y0)
{
	return 0.5 * mPow(y - y0, 2);
}

const Mat nn::CrossEntropy(const Mat &y, const Mat &y0)
{
	return -Mult(y, mLog(y0));
}

const Mat nn::Sigmoid(const Mat &x)
{
	return 1.0 / (1 + mExp(-x));
}

const Mat nn::Tanh(const Mat &x)
{
	return 2 * Sigmoid(2 * x) - 1;
}

const Mat nn::ReLU(const Mat &x)
{
	return mMax(0, x);
}

const Mat nn::ELU(const Mat & x)
{
	Mat x1(x);
	for (int i = 0; i < x.rows(); ++i)
		for (int j = 0; j < x.cols(); ++j)
			for (int z = 0; z < x.channels(); ++z)
				if (x(i, j, z) <= 0)
					x1(i, j, z) = ElU_alpha * (exp(x(i, j, z)) - 1);
	return x1;
}

const Mat nn::SELU(const Mat & x)
{
	return SElU_scale * ELU(x);
}

const Mat nn::LReLU(const Mat & x)
{
	Mat x1(x);
	for (int i = 0; i < x.rows(); ++i)
		for (int j = 0; j < x.cols(); ++j)
			for (int z = 0; z < x.channels(); ++z)
				if (x(i, j, z) < 0)
					x1(i, j, z) *= LReLU_alpha;
	return x1;
}


const Mat nn::D_Softmax(const Mat &y)
{
	return Mult(y, 1 - y);
}

const Mat nn::D_Quadratic(const Mat &y, const Mat &y0)
{
	return y0 - y;
}

const Mat nn::D_CrossEntropy(const Mat &y, const Mat &y0)
{
	return y0 - y;
}

const Mat nn::D_Sigmoid(const Mat &x)
{
	Mat y = Sigmoid(x);
	return Mult(y, 1 - y);
}

const Mat nn::D_Tanh(const Mat &x)
{
	return 4 * D_Sigmoid(x);
}

const Mat nn::D_ReLU(const Mat &x)
{
	Mat x1(x.size3());
	for (int i = 0; i < x.rows(); ++i)
		for (int j = 0; j < x.cols(); ++j)
			for (int z = 0; z < x.channels(); ++z)
				if (x(i, j, z) > 0)
					x1(i, j, z) = 1;
	return x1;
}

const Mat nn::D_ELU(const Mat & x)
{
	Mat x1(x.size3());
	for (int i = 0; i < x.rows(); ++i)
		for (int j = 0; j < x.cols(); ++j)
			for (int z = 0; z < x.channels(); ++z) {
				if (x(i, j, z) > 0)
					x1(i, j, z) = 1;
				else if (x(i, j, z) < 0)
					x1(i, j, z) = ElU_alpha * exp(x(i, j, z));
			}
	return x1;
}

const Mat nn::D_SELU(const Mat & x)
{
	return SElU_scale * D_ELU(x);
}

const Mat nn::D_LReLU(const Mat & x)
{
	Mat x1(x.size3());
	for (int i = 0; i < x.rows(); ++i)
		for (int j = 0; j < x.cols(); ++j)
			for (int z = 0; z < x.channels(); ++z) {
				if (x(i, j, z) > 0)
					x1(i, j, z) = 1;
				else if (x(i, j) < 0)
					x1(i, j, z) = LReLU_alpha;
			}
	return x1;
}

void nn::SetFunc(string func_name, LossFunc *f, LossFunc *df)
{
	if (func_name == "Quadratic") {
		*f = Quadratic;
		*df = D_Quadratic;
	}
	else if (func_name == "CrossEntropy") {
		*f = CrossEntropy;
		*df = D_CrossEntropy;
	}
}
void nn::SetFunc(string func_name, ActivationFunc *f, ActivationFunc *df)
{
	if (func_name == "Softmax") {
		*f = Softmax;
		*df = D_Softmax;
	}
	else if (func_name == "Sigmoid") {
		*f = Sigmoid;
		*df = D_Sigmoid;
	}
	else if (func_name == "Tanh") {
		*f = Tanh;
		*df = D_Tanh;
	}
	else if (func_name == "ReLU") {
		*f = ReLU;
		*df = D_ReLU;
	}
	else if (func_name == "ELU") {
		*f = ELU;
		*df = D_ELU;
	}
	else if (func_name == "SELU") {
		*f = SELU;
		*df = D_SELU;
	}
	else if (func_name == "LReLU") {
		*f = LReLU;
		*df = D_LReLU;
	}
}
void nn::SetFunc(LossFunc func, LossFunc * f, LossFunc * df)
{
	if (func == Quadratic) {
		*f = Quadratic;
		*df = D_Quadratic;
	}
	else if (func == CrossEntropy) {
		*f = CrossEntropy;
		*df = D_CrossEntropy;
	}
}
void nn::SetFunc(ActivationFunc func, ActivationFunc * f, ActivationFunc * df)
{
	if (func == Softmax) {
		*f = Softmax;
		*df = D_Softmax;
	}
	else if (func == Sigmoid) {
		*f = Sigmoid;
		*df = D_Sigmoid;
	}
	else if (func == Tanh) {
		*f = Tanh;
		*df = D_Tanh;
	}
	else if (func == ReLU) {
		*f = ReLU;
		*df = D_ReLU;
	}
	else if (func == ELU) {
		*f = ELU;
		*df = D_ELU;
	}
	else if (func == SELU) {
		*f = SELU;
		*df = D_SELU;
	}
	else if (func == LReLU) {
		*f = LReLU;
		*df = D_LReLU;
	}
}
string nn::Func2String(ActivationFunc f)
{
	string fun_name = "";
	if (f == Softmax) {
		fun_name = "Softmax";
	}
	else if (f == Sigmoid) {
		fun_name = "Sigmoid";
	}
	else if (f == Tanh) {
		fun_name = "Tanh";
	}
	else if (f == ReLU) {
		fun_name = "ReLU";
	}
	else if (f == ELU) {
		fun_name = "ELU";
	}
	else if (f == SELU) {
		fun_name = "SELU";
	}
	else if (f == LReLU) {
		fun_name = "LReLU";
	}
	return fun_name;
}
string nn::Func2String(LossFunc f)
{
	string fun_name = "";
	if (f == Quadratic) {
		fun_name = "Quadratic";
	}
	else if (f == CrossEntropy) {
		fun_name = "CrossEntropy";
	}
	return fun_name;
}
void nn::setfunc(void* train_, LossFunc loss_)
{
	OptimizerP train = (OptimizerP)train_;
	SetFunc(loss_, &train->loss.loss_f, &train->loss.loss_df);
}