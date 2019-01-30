#include <iostream>
#include <fstream>
#include "Train.h"
using namespace nn;
using namespace std;

/*
============================    优化器基类    =========================
*/
Optimizer::Optimizer()
	:step(1e-2), net(nullptr), loss(), method(None) {}
Optimizer::Optimizer(double step) : step(step), net(nullptr), loss(), method(None) {}
Optimizer::~Optimizer() {}
void Optimizer::RegisterNet(Net * net)
{
	this->net = net;
}
bool Optimizer::Enable(const Mat & x, const Mat & y, vector<Mat>& a) const
{
	return !(a.empty() || x.empty() || y.empty() || net == nullptr);
}
void Optimizer::RegisterMethod(OptimizerMethod method)
{
	this->method = method;
}
OptimizerMethod Optimizer::Method() const
{
	return method;
}
/*
========================  注册模型函数(无优化器) ========================
*/
Method::Method() :Optimizer() {}
Method::Method(double step) : Optimizer(step) {}
int Method::Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error) { return 0; }
Optimizer* Method::minimize(LossFunc loss_)const
{
	Optimizer* train = new Method(*this);
	setfunc(train, loss_);
	train->RegisterMethod(None);
	return train;
}
/*
=======================    GradientDescent优化器    =======================
*/
GradientDescentOptimizer::GradientDescentOptimizer(double step) : Optimizer(step) {}
GradientDescentOptimizer::GradientDescentOptimizer(vector<double>& value)
{
	step = value[0];
}
int GradientDescentOptimizer::Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error)
{
	/**
	a = a - step * df(a, x)
	*/
	if (!Enable(x, y, a))return -1;
	int acc = net->Jacobi(x, y, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		dlayer[layer_num] = -step * dlayer[layer_num];
	}
	return acc;
}
Optimizer* GradientDescentOptimizer::minimize(LossFunc loss_)const {
	Optimizer* train = new GradientDescentOptimizer(*this);
	setfunc(train, loss_);
	train->RegisterMethod(GradientDescent);
	return train;
}
Mat GradientDescentOptimizer::data(vector<string> &value_name)const
{
	Mat mat(1, 1);
	mat(0) = step;
	vector<string>(1).swap(value_name);
	value_name[0] = "step";
	return mat;
}
/*
============================    Momentum优化器    =========================
*/
MomentumOptimizer::MomentumOptimizer(double step, double gama) : Optimizer(step), momentum(gama) {}
MomentumOptimizer::MomentumOptimizer(vector<double>& value)
{
	step = value[0];
	momentum = value[1];
}
void MomentumOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	vector<Mat>(size.size()).swap(ma);
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num)
		ma[layer_num] = zeros(size[layer_num]);
}
int MomentumOptimizer::Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error)
{
	/**
	ma = momentum*ma + step * df(a, x)
	a = a - ma
	*/
	if (!Enable(x, y, a))return -1;
	int acc = net->Jacobi(x, y, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		ma[layer_num] = momentum * ma[layer_num] + step * dlayer[layer_num];
		dlayer[layer_num] = -ma[layer_num];
	}
	return acc;
}
Optimizer* MomentumOptimizer::minimize(LossFunc loss_)const {
	Optimizer* train = new MomentumOptimizer(*this);
	setfunc(train, loss_);
	train->RegisterMethod(Momentum);
	return train;
}
Mat MomentumOptimizer::data(vector<string> &value_name)const
{
	Mat mat(2, 1);
	mat(0) = step;
	mat(1) = momentum;
	vector<string>(2).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "momentum";
	return mat;
}
/*
=======================    NesterovMomentum优化器    ======================
*/
NesterovMomentumOptimizer::NesterovMomentumOptimizer(double step, double gama) : Optimizer(step), momentum(gama) {}
NesterovMomentumOptimizer::NesterovMomentumOptimizer(vector<double>& value)
{
	step = value[0];
	momentum = value[1];
}
void NesterovMomentumOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	vector<Mat>(size.size()).swap(ma);
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num)
		ma[layer_num] = zeros(size[layer_num]);
}
int NesterovMomentumOptimizer::Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error)
{
	/**
	ma = momentum*ma + step * df(a - momentum*ma, x)
	a = a - ma
	*/
	if (!Enable(x, y, a))return -1;
	vector<Mat>(net->LayerNum()).swap(dlayer);
	for (size_t layer_num = 0; layer_num < net->LayerNum(); ++layer_num) {
		dlayer[layer_num] = -momentum * ma[layer_num];
	}
	int acc = net->FutureJacobi(x, y, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		ma[layer_num] = momentum * ma[layer_num] + step * dlayer[layer_num];
		dlayer[layer_num] = -ma[layer_num];
	}
	return acc;
}
Optimizer* NesterovMomentumOptimizer::minimize(LossFunc loss_)const {
	Optimizer* train = new NesterovMomentumOptimizer(*this);
	setfunc(train, loss_);
	train->RegisterMethod(NesterovMomentum);
	return train;
}
Mat NesterovMomentumOptimizer::data(vector<string> &value_name)const
{
	Mat mat(2, 1);
	mat(0) = step;
	mat(1) = momentum;	
	vector<string>(2).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "momentum";
	return mat;
}
/*
=============================    Adagrad优化器    ============================
*/
AdagradOptimizer::AdagradOptimizer(double step, double epsilon) : Optimizer(step), epsilon(epsilon) {}
AdagradOptimizer::AdagradOptimizer(vector<double>& value)
{
	step = value[0];
	epsilon = value[1];
}
void AdagradOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	vector<Mat>(size.size()).swap(alpha);
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num)
		alpha[layer_num] = zeros(size[layer_num]);
}
int AdagradOptimizer::Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error)
{
	/**
	alpha = alpha + df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	if (!Enable(x, y, a))return -1;
	int acc = net->Jacobi(x, y, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		alpha[layer_num] = alpha[layer_num] + mPow(dlayer[layer_num], 2);
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), dlayer[layer_num]);
	}
	return acc;
}
Optimizer * AdagradOptimizer::minimize(LossFunc loss_) const
{
	Optimizer* train = new AdagradOptimizer(*this);
	setfunc(train, loss_);
	train->RegisterMethod(Adagrad);
	return train;
}
Mat AdagradOptimizer::data(vector<string> &value_name)const
{
	Mat mat(2, 1);
	mat(0) = step;
	mat(1) = epsilon;	
	vector<string>(2).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "epsilon";
	return mat;
}
/*
=============================    RMSProp优化器    ============================
*/
RMSPropOptimizer::RMSPropOptimizer(double step, double decay, double epsilon) : Optimizer(step), decay(decay), epsilon(epsilon) {}
RMSPropOptimizer::RMSPropOptimizer(vector<double>& value)
{
	step = value[0];
	decay = value[1];
	epsilon = value[2];
}
void RMSPropOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	vector<Mat>(size.size()).swap(alpha);
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num)
		alpha[layer_num] = zeros(size[layer_num]);
}
int RMSPropOptimizer::Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error)
{
	/**
	alpha = beta*alpha + (1 - beta)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	if (!Enable(x, y, a))return -1;
	int acc = net->Jacobi(x, y, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		alpha[layer_num] = decay * alpha[layer_num] + (1 - decay) *  mPow(dlayer[layer_num], 2);
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), dlayer[layer_num]);
	}
	return acc;
}
Optimizer * RMSPropOptimizer::minimize(LossFunc loss_) const
{
	Optimizer* train = new RMSPropOptimizer(*this);
	setfunc(train, loss_);
	train->RegisterMethod(RMSProp);
	return train;
}
Mat RMSPropOptimizer::data(vector<string> &value_name)const
{
	Mat mat(3, 1);
	mat(0) = step;
	mat(1) = decay;
	mat(2) = epsilon;
	vector<string>(3).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "decay";
	value_name[2] = "epsilon";
	return mat;
}
/*
=============================    Adam优化器    ============================
*/
AdamOptimizer::AdamOptimizer(double step, double beta1, double beta2, double epsilon)
	: beta1(beta1), beta2(beta2), epsilon(epsilon), Optimizer(step), ma(), alpha() {}
AdamOptimizer::AdamOptimizer(vector<double>& value) : ma(), alpha()
{
	step = value[0];
	beta1 = value[1];
	beta2 = value[2];
	epsilon = value[3];
}
void AdamOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	vector<Mat>(size.size()).swap(ma);
	vector<Mat>(size.size()).swap(alpha);
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num) {
		ma[layer_num] = zeros(size[layer_num]);
		alpha[layer_num] = zeros(size[layer_num]);
	}
}
int AdamOptimizer::Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error)
{
	/**
	ma = beta1*ma + (1 - beta1)*df(a, x)
	alpha = beta2*alpha + (1 - beta2)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	if (!Enable(x, y, a))return -1;
	int acc = net->Jacobi(x, y, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		Mat d = dlayer[layer_num];
		ma[layer_num] = beta1 * ma[layer_num] + (1 - beta1)*dlayer[layer_num];
		alpha[layer_num] = beta2 * alpha[layer_num] + (1 - beta2)*mPow(dlayer[layer_num], 2);
		//ma[layer_num] /= (1 - beta1);
		//alpha[layer_num] /= (1 - beta2);
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), ma[layer_num]);
	}
	return acc;
}
Optimizer * AdamOptimizer::minimize(LossFunc loss_) const
{
	Optimizer* train = new AdamOptimizer(*this);
	setfunc(train, loss_);
	train->RegisterMethod(Adam);
	return train;
}
Mat AdamOptimizer::data(vector<string> &value_name)const
{
	Mat mat(4, 1);
	mat(0) = step;
	mat(1) = beta1;
	mat(2) = beta2;
	mat(3) = epsilon;
	vector<string>(4).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "beta1";
	value_name[2] = "beta2";
	value_name[3] = "epsilon";
	return mat;
}
/*
=============================    NesterovAdam优化器    ============================
*/
NesterovAdamOptimizer::NesterovAdamOptimizer(double step, double beta1, double beta2, double epsilon)
	: beta1(beta1), beta2(beta2), epsilon(epsilon), Optimizer(step), ma(), alpha() {}
NesterovAdamOptimizer::NesterovAdamOptimizer(vector<double>& value) : ma(), alpha()
{
	step = value[0];
	beta1 = value[1];
	beta2 = value[2];
	epsilon = value[3];
}
void NesterovAdamOptimizer::init(vector<Size3>& size)
{
	if (size.empty())return;
	vector<Mat>(size.size()).swap(ma);
	vector<Mat>(size.size()).swap(alpha);
	for (size_t layer_num = 0; layer_num < size.size(); ++layer_num) {
		ma[layer_num] = zeros(size[layer_num]);
		alpha[layer_num] = zeros(size[layer_num]);
	}
}
int NesterovAdamOptimizer::Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error)
{
	/**
	ma = beta1*ma + (1 - beta1)*df(a - step/sqrt(alpha + epsilon)*ma, x)
	alpha = beta2*alpha + (1 - beta2)*df(a - step/sqrt(alpha + epsilon)*ma, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	if (!Enable(x, y, a))return -1;
	vector<Mat>(net->LayerNum()).swap(dlayer);
	for (size_t layer_num = 0; layer_num < net->LayerNum(); ++layer_num) {
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), ma[layer_num]);
	}
	int acc = net->FutureJacobi(x, y, dlayer, error);
	for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
		Mat d = dlayer[layer_num];
		ma[layer_num] = beta1 * ma[layer_num] + (1 - beta1)*dlayer[layer_num];
		alpha[layer_num] = beta2 * alpha[layer_num] + (1 - beta2)*mPow(dlayer[layer_num], 2);
		dlayer[layer_num] = -Mult(step / mSqrt(alpha[layer_num] + epsilon), ma[layer_num]);
	}
	return acc;
}
Optimizer * NesterovAdamOptimizer::minimize(LossFunc loss_) const
{
	Optimizer* train = new NesterovAdamOptimizer(*this);
	setfunc(train, loss_);
	train->RegisterMethod(NesterovAdam);
	return train;
}

Mat NesterovAdamOptimizer::data(vector<string> &value_name)const
{
	Mat mat(4, 1);
	mat(0) = step;
	mat(1) = beta1;
	mat(2) = beta2;
	mat(3) = epsilon;
	vector<string>(4).swap(value_name);
	value_name[0] = "step";
	value_name[1] = "beta1";
	value_name[2] = "beta2";
	value_name[3] = "epsilon";
	return mat;
}
