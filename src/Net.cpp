#include <iostream>
#include <fstream>
#include "include.h"
using namespace std;
using namespace nn;

Net::Net() : op(nullptr)
{
	ClearLayer();
}
Net::Net(Optimizer* op)
{
	ClearLayer();
	InitMethod(op);
}
Net::Net(OptimizerMethod opm, double step, LossFunc loss_f)
{
	switch (opm)
	{
	case None:
		op = Method().minimize(loss_f);
		break;
	case GradientDescent:
		op = GradientDescentOptimizer(step).minimize(loss_f);
		break;
	case Momentum:
		op = MomentumOptimizer(step).minimize(loss_f);
		break;
	case NesterovMomentum:
		op = NesterovMomentumOptimizer(step).minimize(loss_f);
		break;
	case Adagrad:
		op = AdagradOptimizer(step).minimize(loss_f);
		break;
	case RMSProp:
		op = RMSPropOptimizer(step).minimize(loss_f);
		break;
	case Adam:
		op = AdamOptimizer(step).minimize(loss_f);
		break;
	case NesterovAdam:
		op = NesterovAdamOptimizer(step).minimize(loss_f);
		break;
	default:
		break;
	}
}
Net::Net(string model_path) : op(nullptr)
{
	ClearLayer();
	ReadModel(model_path);
}
Net::~Net()
{
	ClearLayer();
	if (op != nullptr) {
		delete op;
		op = nullptr;
	}
}

void Net::ClearLayer()
{
	vector<LayerType>().swap(config);
	vector<ConvInfo>().swap(convinfo);
	vector<Size>().swap(poolinfo);
	vector<Mat>().swap(layer);
	vector<Size3>().swap(shape);
	vector<Activation>().swap(activation);
}
void Net::AddDropout(double dropout_prob)
{
	if (dropout_prob < 0 || dropout_prob > 1) {
		fprintf(stderr, "dropout_prob must be between zero and one!");
		throw dropout_prob;
	}
	config.push_back(DROPOUT);
	dropout.push_back(dropout_prob);
}
void Net::AddActivation(ActivationFunc act_f)
{
	config.push_back(ACTIVATION);
	Activation act;
	SetFunc(act_f, &act.activation_f, &act.activation_df);
	activation.push_back(act);
}
void Net::AddMaxPool(Size poolsize)
{
	config.push_back(MAX_POOL);
	poolinfo.push_back(poolsize);
}
void Net::AddMaxPool(int pool_row, int pool_col)
{
	config.push_back(MAX_POOL);
	poolinfo.push_back(Size(pool_row, pool_col));
}
void Net::AddAveragePool(Size poolsize)
{
	config.push_back(AVERAGE_POOL);
	poolinfo.push_back(poolsize);
}
void Net::AddAveragePool(int pool_row, int pool_col)
{
	config.push_back(AVERAGE_POOL);
	poolinfo.push_back(Size(pool_row, pool_col));
}
void Net::AddReshape(Size re_size)
{
	config.push_back(RESHAPE);
	shape.push_back(Size3(re_size.hei, re_size.wid));
}
void Net::AddReshape(Size3 re_size)
{
	config.push_back(RESHAPE);
	shape.push_back(re_size);
}
void Net::AddReshape(int row, int col, int channel)
{
	config.push_back(RESHAPE);
	shape.push_back(Size3(row, col, channel));
}
void Net::AddFullConnect(const Mat & full_layer, const Mat & full_bias)
{
	if (full_bias.cols() != 1) {
		fprintf(stderr, "bias's col must be one!");
		throw full_bias;
	}
	if (full_layer.cols() != full_bias.rows()) {
		fprintf(stderr, "layer's col must be the same as bias's row!");
		throw full_layer;
	}
	else {
		layer.push_back(full_layer.Tran());
		layer.push_back(full_bias);
		config.push_back(FULL_CONNECTION);
	}
}
void Net::AddFullConnect(int layer_row, int layer_col, int bias_row, int bias_col)
{
	if (bias_col != 1) {
		fprintf(stderr, "bias's col must be one!");
		throw bias_col;
	}
	if (layer_col != bias_row) {
		fprintf(stderr, "layer's col must be the same as bias's row!");
		throw layer_row;
	}
	else {
		layer.push_back(CreateMat(layer_col, layer_row));
		layer.push_back(CreateMat(bias_row, bias_col));
		config.push_back(FULL_CONNECTION);
	}
}
void Net::AddFullConnect(int layer_row, int layer_col, int layer_channel, int bias_row, int bias_col, int bias_channel)
{
	if (bias_col != 1) {
		fprintf(stderr, "bias's col must be one!");
		throw bias_col;
	}
	if (layer_col != bias_row) {
		fprintf(stderr, "layer's col must be the same as bias's row!");
		throw layer_row;
	}
	else {
		layer.push_back(CreateMat(layer_col, layer_row, layer_channel));
		layer.push_back(CreateMat(bias_row, bias_col, bias_channel));
		config.push_back(FULL_CONNECTION);
	}
}
void Net::AddConv(const Mat & conv_layer, const Mat & conv_bias, ConvInfo info)
{
	if (conv_bias.cols() != 1|| conv_bias.rows() != 1) {
		fprintf(stderr, "bias's row and col must be one!");
		throw conv_bias;
	}
	config.push_back(CONV2D);
	convinfo.push_back(info);
	layer.push_back(conv_layer);
	layer.push_back(conv_bias);
}
void Net::AddConv(const Mat & conv_layer, const Mat & conv_bias, bool is_copy_border, Size strides, Point anchor)
{
	if (conv_bias.cols() != 1 || conv_bias.rows() != 1) {
		fprintf(stderr, "bias's row and col must be one!");
		throw conv_bias;
	}
	config.push_back(CONV2D);
	convinfo.push_back(ConvInfo(strides, anchor, is_copy_border));
	layer.push_back(conv_layer);
	layer.push_back(conv_bias);
}
void Net::AddConv(int kern_row, int kern_col, int input_channel, int output_channel, bool is_copy_border, Size strides, Point anchor)
{
	config.push_back(CONV2D);
	convinfo.push_back(ConvInfo(strides, anchor, is_copy_border));
	layer.push_back(CreateMat(kern_row, kern_col, output_channel*input_channel));
	layer.push_back(CreateMat(1, 1, output_channel));
}

void Net::InitModel(NetConfig & config_)
{
	if (config_.op != nullptr)
		InitMethod(config_.op);
	if (!config_.layer.empty())
		SetLayer(config_.layer);
	if (!config_.config.empty())
		SetLayerType(config_.config);
	if (!config_.convinfo.empty())
		SetConvinfo(config_.convinfo);
	if (!config_.poolinfo.empty())
		SetPoolinfo(config_.poolinfo);
	if (!config_.shape.empty())
		SetReshape(config_.shape);
	if (!config_.activation.empty())
		SetActivation(config_.activation);
	if (!config_.dropout.empty())
		SetDropout(config_.dropout);
}
void Net::SetLayer(vector<Mat>& layer)
{
	layer.swap(this->layer);
}
void Net::SetLayerType(vector<LayerType>& config)
{
	config.swap(this->config);
}
void Net::SetConvinfo(vector<ConvInfo>& convinfo)
{
	convinfo.swap(this->convinfo);
}
void Net::SetPoolinfo(vector<Size>& poolinfo)
{
	poolinfo.swap(this->poolinfo);
}
void Net::SetReshape(vector<Size3>& shape)
{
	shape.swap(this->shape);
}
void Net::SetActivation(vector<Activation>& activation)
{
	activation.swap(this->activation);
}
void Net::SetDropout(vector<double>& dropout)
{
	dropout.swap(this->dropout);
}

int Net::TrainModel(const Mat & input, const Mat & label, double *error)
{
	if (op == nullptr)return -1;
	if (input.empty() || label.empty())return -1;
	vector<Mat> dlayer;
	double err; 
	int acc;
	if (error == nullptr) 
		acc = BackPropagation(input, label, dlayer, err);
	else
		acc = BackPropagation(input, label, dlayer, *error);
	for (size_t layer_num = 0; layer_num < layer.size(); ++layer_num) {
		layer[layer_num] += dlayer[layer_num];
	}
	return acc == 1 ? 1 : 0;
}
double Net::TrainModel(const vector<Mat> &input, const vector<Mat> &label, double *error)
{
	if (op == nullptr)return NAN;
	if (input.empty() || label.empty())return NAN;
	vector<Mat> dlayer(layer.size());
	for (size_t layer_num = 0; layer_num < layer.size(); ++layer_num) {
		dlayer[layer_num] = Mat(layer[layer_num].size3());
	}
	int acc = 0;
	double err = 0;
	double error_ = 0;
	for (vector<Mat>::const_iterator x = input.begin(), y = label.begin();
		x != input.end(), y != label.end(); ++x, ++y) {
		vector<Mat> d_layer;
		acc += BackPropagation(*x, *y, d_layer, err);	
		error_ += err;
		for (size_t layer_num = 0; layer_num < dlayer.size(); ++layer_num) {
			dlayer[layer_num] += d_layer[layer_num];
		}
	}
	error_ /= (double)input.size();
	if (error != nullptr)
		*error = error_;
	double mean_acc = acc / (double)input.size();
	//if (mean_acc != 1) {
		for (size_t layer_num = 0; layer_num < layer.size(); ++layer_num) {
			layer[layer_num] += dlayer[layer_num] / (double)input.size();
		}
	//}
	return mean_acc;
}
Mat Net::Run(const Mat & input) const
{
	if (op == nullptr)return Mat();
	if (layer.empty())return Mat();
	if (config.empty())return Mat();
	Mat y = input;
	for (size_t 
		index = 0, 
		layer_num = 0,
		conv_layer = 0,
		pool_layer = 0,
		shape_layer = 0,
		activation_layer = 0;
		index < config.size() && layer_num < layer.size(); ++index) {
		switch (config[index])
		{
		case CONV2D:
			y = conv2d(y, layer[layer_num], convinfo[conv_layer].strides, convinfo[conv_layer].anchor, convinfo[conv_layer].is_copy_border) + layer[layer_num + 1];
			conv_layer += 1;
			layer_num += 2;
			break;
		case MAX_POOL:
			y = MaxPool(y, poolinfo[pool_layer]);
			pool_layer += 1;
			break;
		case AVERAGE_POOL:
			y = AveragePool(y, poolinfo[pool_layer]);
			pool_layer += 1;
			break;
		case FULL_CONNECTION:
			y = FullConnection(y, layer[layer_num], layer[layer_num + 1]);
			layer_num += 2;
			break;
		case ACTIVATION:
			y = activation[activation_layer].activation_f(y);
			activation_layer += 1;
			break;
		case RESHAPE:
			y.reshape(shape[shape_layer].x, shape[shape_layer].y, shape[shape_layer].z);
			shape_layer += 1;
		case DROPOUT:break;
		default:
			break;
		}
	}
	return y;
}
void Net::ForwardPropagation(const Mat & input, Mat & output, const vector<Mat> & layer, vector<Mat>& variable) const
{
	Mat mark;
	Mat y = input;
	vector<Mat>().swap(variable);
	variable.push_back(input);
	for (size_t
		index = 0,
		layer_num = 0,
		conv_layer = 0,
		pool_layer = 0,
		shape_layer = 0,
		dropout_layer = 0,
		activation_layer = 0;
		index < config.size(); ++index) {
		switch (config[index])
		{
		case CONV2D:
			y = conv2d(y, layer[layer_num], convinfo[conv_layer].strides, convinfo[conv_layer].anchor, convinfo[conv_layer].is_copy_border) + layer[layer_num + 1];
			variable.push_back(y);
			conv_layer += 1;
			layer_num += 2;
			break;
		case MAX_POOL: 
			y = MaxPool(y, mark, poolinfo[pool_layer]);
			variable.push_back(mark);
			variable.push_back(y);
			pool_layer += 1;
			break;
		case AVERAGE_POOL:
			y = AveragePool(y, poolinfo[pool_layer]);
			variable.push_back(y);
			pool_layer += 1;
			break;
		case FULL_CONNECTION:					
			y = FullConnection(y, layer[layer_num], layer[layer_num + 1]);
			variable.push_back(y);
			layer_num += 2;
			break;
		case ACTIVATION:
			y = activation[activation_layer].activation_f(y);
			activation_layer += 1;
			break;
		case RESHAPE: 
			{
			Mat mat(3, 1, 1);
			mat(0) = y.rows();
			mat(1) = y.cols();
			mat(2) = y.channels();
			variable.push_back(mat);
			y.reshape(shape[shape_layer].x, shape[shape_layer].y, shape[shape_layer].z);
			variable.push_back(y);
			shape_layer += 1;
			}
			break;
		case DROPOUT: 
			if (dropout[dropout_layer] != 0) {
				Mat drop = mThreshold(mRand(0, 1, y.rows(), y.cols(), y.channels(), true), dropout[dropout_layer], 0, 1);
				y = Mult(y, drop);
				y *= 1 / (1 - dropout[dropout_layer]);
			}
			dropout_layer += 1;
			break;
		default:
			break;
		}
	}
	variable.push_back(y);
	y.swap(output);
}
int Net::BackPropagation(const Mat & input, const Mat & output, vector<Mat>& d_layer, double & error)
{
	return op->Run(d_layer, input, output, layer, error);
}
int Net::FutureJacobi(const Mat & input, const Mat & output, vector<Mat>& d_layer, double & error) const
{
	vector<Mat> layer_(layer);
	for (size_t index = 0; index < layer.size(); ++index) {
		layer_[index] += d_layer[index];
	}
	vector<Mat> variable;
	Mat transmit;
	ForwardPropagation(input, transmit, layer_, variable);
	error = op->loss.loss_f(output, transmit).Norm(2);
	JacobiMat(d_layer, variable, output, transmit);
	if (output.maxAt() == transmit.maxAt())return 1;
	else return  0;
}
int Net::Jacobi(const Mat & input, const Mat & output, vector<Mat>& d_layer, double & error) const
{
	vector<Mat> variable;
	Mat transmit;
	ForwardPropagation(input, transmit, layer, variable);
	error = op->loss.loss_f(output, transmit).Norm(2);
	JacobiMat(d_layer, variable, output, transmit);
	if (output.maxAt() == transmit.maxAt())return 1;
	else return  0;
}
void Net::JacobiMat(
	vector<Mat> &d_layer, const vector<Mat> &x,
	const Mat &y, const Mat &output)const
{
	Mat value;
	vector<Mat> dlayer(layer.size());
	value = op->loss.loss_df(y, output);
	int size = (int)config.size();
	for (int
		number = 0,
		index = (int)config.size() - 1,
		layer_num = (int)layer.size() - 1,
		x_num = (int)x.size() - 1,
		conv_layer = (int)convinfo.size() - 1,
		pool_layer = (int)poolinfo.size() - 1,
		activation_layer = (int)activation.size() - 1;
		index >= 0 && layer_num >= 0; --index) {
		switch (config[index]){
		case CONV2D:
			dlayer[layer.size() - 1 - number] = mSum(value, CHANNEL);
			dlayer[layer.size() - 2 - number] = iconv2d(x[x_num - 1], value, layer[layer_num - 1].size3(), convinfo[conv_layer], false);
			number += 2;
			if (index > 0 && layer_num > 1)
				value = iconv2d(value, layer[layer_num - 1].Tran(), Size3(), convinfo[conv_layer], true);
			conv_layer -= 1;
			layer_num -= 2;
			x_num -= 1;
			break;
		case MAX_POOL:
			value = upsample(value, poolinfo[pool_layer], x[x_num - 1]);
			pool_layer -= 1;
			x_num -= 2;
			break;
		case AVERAGE_POOL:
			value = upsample(value, poolinfo[pool_layer]);
			pool_layer -= 1;
			break;
		case FULL_CONNECTION:
			dlayer[layer.size() - 1 - number] = value;
			dlayer[layer.size() - 2 - number] = value * x[x_num - 1].Tran();
			number += 2;
			if (index >= 0 && layer_num >= 0)
				value = layer[layer_num - 1].Tran() * value;
			layer_num -= 2;
			x_num -= 1;
			break;
		case ACTIVATION:
			if (index == config.size() - 1) {
				if(op->loss.loss_f != CrossEntropy)
					value = Mult(value, activation[activation_layer].activation_df(x[x_num]));	
				x_num -= 1;
			}
			else {
				value = Mult(value, activation[activation_layer].activation_df(x[x_num]));
			}	
			activation_layer -= 1;
			break;
		case RESHAPE:
			if (x[x_num].length() != 3)x_num -= 1;
			value.reshape((int)x[x_num](0), (int)x[x_num](1), (int)x[x_num](2));
			x_num -= 1;
		case DROPOUT:break;
		default:
			break;
		}
	}
	dlayer.swap(d_layer);
}

void Net::InitMethod(Optimizer* optrimizer)
{
	if (op != nullptr) {
		delete op;
		op = nullptr;
	}
	if (op == nullptr) {
		op = optrimizer;
	}
	if (layer.empty())return;
	vector<Size3> size;
	for (size_t layer_num = 0; layer_num < layer.size(); ++layer_num) {
		size.push_back(layer[layer_num].size3());
	}
	optrimizer->init(size);
	optrimizer->RegisterNet((Net*)this);
}
void Net::ReadModel(string model_path)
{
	int data_num;
	double data_;
	string str;
	string lossname;
	vector<double> optimizer_data;
	OptimizerP optimizer = nullptr;

	ifstream model(model_path, ios::binary);
	if (!model.is_open())goto MODEL_OPEN_FAIL;
	getline(model, str);
	if (str != "use optimizer method")goto MODEL_READ_FAIL;
	model >> str;
	if (str != "loss_function")goto MODEL_READ_FAIL;
	model >> lossname;
	model >> str;
	if (str != "optimizer_data")goto MODEL_READ_FAIL;
	model >> data_num;
	vector<double>(data_num).swap(optimizer_data);
	for (int i = 0; i < data_num; i++) {
		model >> str;
		model >> data_;
		optimizer_data[i] = data_;
	}
	model.ignore();
	getline(model, str);
	str = str.substr(0, str.find(':'));
	if (str == "None") optimizer = Method().minimize();
	else if (str == "GradientDescent") optimizer = GradientDescentOptimizer(optimizer_data).minimize();
	else if (str == "Momentum") optimizer = MomentumOptimizer(optimizer_data).minimize();
	else if (str == "NesterovMomentum")optimizer = NesterovMomentumOptimizer(optimizer_data).minimize();
	else if (str == "Adagrad")optimizer = AdagradOptimizer(optimizer_data).minimize();
	else if (str == "RMSProp") optimizer = RMSPropOptimizer(optimizer_data).minimize();
	else if (str == "Adam") optimizer = AdamOptimizer(optimizer_data).minimize();
	else if (str == "NesterovAdam") optimizer = NesterovAdamOptimizer(optimizer_data).minimize();
	else goto MODEL_READ_FAIL;
	SetFunc(lossname, &optimizer->loss.loss_f, &optimizer->loss.loss_df);
	InitMethod(optimizer);

	model >> str;
	if (str != "layer_size")goto MODEL_READ_FAIL;
	int config_num;
	model >> config_num;
	for (int i = 0; i < config_num; i++) {
		model >> str;
		model >> str;
		model >> str;
		if (str == "CONV2D") {
			bool copy_border;
			Size strides;
			Point anchor;
			model >> str;
			if (str != "strides")goto MODEL_READ_FAIL;
			model >> str;
			if (str != "strides_x")goto MODEL_READ_FAIL;
			model >> strides.hei;
			model >> str;
			if (str != "strides_y")goto MODEL_READ_FAIL;
			model >> strides.wid;
			model >> str;
			if (str != "anchor")goto MODEL_READ_FAIL;
			model >> str;
			if (str != "anchor_x")goto MODEL_READ_FAIL;
			model >> anchor.x;
			model >> str;
			if (str != "anchor_y")goto MODEL_READ_FAIL;
			model >> anchor.y;
			model >> str;
			if (str != "padding")goto MODEL_READ_FAIL;
			model >> str;
			if (str == "SAME") copy_border = true;
			else if (str == "VALID") copy_border = false;
			else goto MODEL_READ_FAIL;
			AddConv(copy_border, strides, anchor);
		}
		else if (str == "MAX_POOL") {
			Size pool_size;
			model >> str;
			if (str != "pool_size")goto MODEL_READ_FAIL;
			model >> str;
			if (str != "row")goto MODEL_READ_FAIL;
			model >> pool_size.hei;
			model >> str;
			if (str != "col")goto MODEL_READ_FAIL;
			model >> pool_size.wid;
			AddMaxPool(pool_size);
		}
		else if (str == "AVERAGE_POOL") {
			Size pool_size;
			model >> str;
			if (str != "pool_size")goto MODEL_READ_FAIL;
			model >> str;
			if (str != "row")goto MODEL_READ_FAIL;
			model >> pool_size.hei;
			model >> str;
			if (str != "col")goto MODEL_READ_FAIL;
			model >> pool_size.wid;
			AddAveragePool(pool_size);
		}
		else if (str == "FULL_CONNECTION") {
			config.push_back(FULL_CONNECTION);
		}
		else if (str == "ACTIVATION") {
			model >> str;
			if (str != "activation_function")goto MODEL_READ_FAIL;
			model >> str;
			Activation fun;
			SetFunc(str, &fun.activation_f, &fun.activation_df);
			config.push_back(ACTIVATION);
			activation.push_back(fun);
		}
		else if (str == "RESHAPE") {
			Size3 reshape_size;
			model >> str;
			if (str != "reshape_size")goto MODEL_READ_FAIL;
			model >> str;
			if (str != "reshape_row")goto MODEL_READ_FAIL;
			model >> reshape_size.x;
			model >> str;
			if (str != "reshape_col")goto MODEL_READ_FAIL;
			model >> reshape_size.y;
			model >> str;
			if (str != "reshape_channel")goto MODEL_READ_FAIL;
			model >> reshape_size.z;
			AddReshape(reshape_size);
		}
		else if (str == "DROPOUT") {
			double dropout_p;
			model >> str;
			if (str != "dropout")goto MODEL_READ_FAIL;
			model >> dropout_p;
			AddDropout(dropout_p);
		}
		else goto MODEL_READ_FAIL;
	}

	int layer_num;
	model >> str;
	if (str != "param_num")goto MODEL_READ_FAIL;
	model >> layer_num;
	vector<Mat>(layer_num * 2).swap(layer);
	model >> str;
	if (str != "param_size")goto MODEL_READ_FAIL;
	for (size_t index = 0; index < layer.size(); ++index) {
		int layer_row;
		int layer_col;
		int layer_channel;
		model >> layer_row;
		model >> layer_col;
		model >> layer_channel;
		layer[index] = zeros(layer_row, layer_col, layer_channel);
	}
	model >> str;
	if (str != "param_data")goto MODEL_READ_FAIL;
	for (size_t index = 0; index < layer.size(); ++index) {
		for (int i = 0; i < layer[index].rows(); ++i) {
			for (int j = 0; j < layer[index].cols(); ++j) {
				for (int z = 0; z < layer[index].channels(); ++z) {
					model >> layer[index](i, j, z);
				}
			}
		}
	}
	model.close();

	return;
MODEL_READ_FAIL:
	if (optimizer != nullptr) {
		delete optimizer;
		optimizer = nullptr;
	}
	model.close();
MODEL_OPEN_FAIL:
	ClearLayer();
	fprintf(stderr, "error: load %s fail!\n", model_path.c_str());
}
void Net::SaveModel(string save_path) const
{
	//if (0 != _access(save_path.c_str(), 0))
	//{
	//	// if this folder not exist, create a new one.
	//	_mkdir(save_path.c_str());
	//}
	Mat data; vector<string> data_name; string method_opt;
	ofstream model(save_path, ios::binary);
	if (model.is_open())
	{
		model << "use optimizer method" << endl;
		model << "loss_function " << Func2String(op->loss.loss_f) << endl;
		switch (op->Method())
		{
		case None:
			method_opt = "None: Support only Forward propagation !";
			break;
		case GradientDescent:
			method_opt = "GradientDescent: Support forward propagation and back propagation";
			data = ((GradientDescentOptimizer*)op)->data(data_name);
			break;
		case Momentum:
			method_opt = "Momentum: Support forward propagation and back propagation";
			data = ((MomentumOptimizer*)op)->data(data_name);
			break;
		case NesterovMomentum:
			method_opt = "NesterovMomentum: Support forward propagation and back propagation";
			data = ((NesterovMomentumOptimizer*)op)->data(data_name);
			break;
		case Adagrad:
			method_opt = "Adagrad: Support forward propagation and back propagation";
			data = ((AdagradOptimizer*)op)->data(data_name);
			break;
		case RMSProp:
			method_opt = "RMSProp: Support forward propagation and back propagation";
			data = ((RMSPropOptimizer*)op)->data(data_name);
			break;
		case Adam:
			method_opt = "Adam: Support forward propagation and back propagation";
			data = ((AdamOptimizer*)op)->data(data_name);
			break;
		case NesterovAdam:
			method_opt = "NesterovAdam: Support forward propagation and back propagation";
			data = ((NesterovAdamOptimizer*)op)->data(data_name);
			break;
		default:
			break;
		}
		model << "optimizer_data " << data.length() << endl;
		for (int i = 0; i < data.length(); i++)
			model << data_name[i] << ' ' << data(i) << endl;
		model << method_opt << endl;


		model << "layer_size" << endl;
		model << config.size() << endl;
		for (size_t  i = 0,
			conv_layer = 0,
			pool_layer = 0,
			shape_layer = 0,
			dropout_layer = 0,
			activation_layer = 0; i < config.size(); ++i) {
			model << "layer_num  " << "No." << i + 1 << endl;
			switch (config[i])
			{
			case CONV2D:
				model << "CONV2D" << endl;
				model << "strides" << endl;
				model << "strides_x " << convinfo[conv_layer].strides.hei << endl;
				model << "strides_y " << convinfo[conv_layer].strides.wid << endl;
				model << "anchor" << endl;
				model << "anchor_x " << convinfo[conv_layer].anchor.x << endl;
				model << "anchor_y " << convinfo[conv_layer].anchor.x << endl;
				model << "padding " << (convinfo[conv_layer].is_copy_border ? "SAME" : "VALID") << endl;
				conv_layer += 1;
				break;
			case MAX_POOL:
				model << "MAX_POOL" << endl;
				model << "pool_size" << endl;
				model << "row " << poolinfo[pool_layer].hei << endl;
				model << "col " << poolinfo[pool_layer].wid << endl;
				pool_layer += 1;
				break;
			case AVERAGE_POOL:
				model << "AVERAGE_POOL" << endl;
				model << "pool_size" << endl;
				model << "row " << poolinfo[pool_layer].hei << endl;
				model << "col " << poolinfo[pool_layer].wid << endl;
				pool_layer += 1;
				break;
			case FULL_CONNECTION:
				model << "FULL_CONNECTION" << endl;
				break;
			case ACTIVATION:
				model << "ACTIVATION" << endl;
				model << "activation_function " << Func2String(activation[activation_layer].activation_f) << endl;
				activation_layer += 1;
				break;
			case RESHAPE:
				model << "RESHAPE" << endl;
				model << "reshape_size" << endl;
				model << "reshape_row " << shape[shape_layer].x << endl;
				model << "reshape_col " << shape[shape_layer].y << endl;
				model << "reshape_channel " << shape[shape_layer].z << endl;
				shape_layer += 1;
				break;
			case DROPOUT:
				model << "DROPOUT" << endl;
				model << "dropout " << dropout[dropout_layer] << endl;
				dropout_layer += 1;
			default:
				break;
			}
		}

		model << "param_num" << endl;
		model << layer.size() / 2 << endl;
		model << "param_size" << endl;
		for (size_t index = 0; index < layer.size(); ++index) {
			model << layer[index].rows() << ' ' << layer[index].cols() << ' ' << layer[index].channels() << endl;
		}
		model << "param_data" << endl;
		for (size_t index = 0; index < layer.size(); ++index) {
			for (int i = 0; i < layer[index].rows(); ++i) {
				for (int j = 0; j < layer[index].cols(); ++j) {
					for (int z = 0; z < layer[index].channels(); ++z) {
						model << layer[index](i, j, z) << ' ';
					}
				}
				model << endl;
			}
		}

		model.close();
	}
}
size_t Net::LayerNum() const
{
	return layer.size();
} 
bool Net::Run(const Mat & input, const Mat & output) const
{
	return (Run(input).maxAt() == output.maxAt());
}
double Net::Accuracy(const vector<Mat>& input, const vector<Mat>& output) const
{
	if (input.empty() || output.empty())return NAN;
	if (input.size() != output.size())return NAN;
	int success = 0;
	int sum = 0;
	for (size_t index = 0; index < input.size(); ++index) {
		if (Run(input[index], output[index]))
			success++;
		sum++;
	}
	return (double)success / (double)sum;
}
void Net::AddConv(bool is_copy_border, Size strides, Point anchor)
{
	config.push_back(CONV2D);
	convinfo.push_back(ConvInfo(strides, anchor, is_copy_border));
}

Optimizer * nn::CreateOptimizer(OptimizerMethod opm, double step, LossFunc loss_f, const Mat & value)
{
	switch (opm)
	{
	case None:
		return Method().minimize(loss_f);
	case GradientDescent:
		return GradientDescentOptimizer(step).minimize(loss_f);
	case Momentum:
		return MomentumOptimizer(step, value(0)).minimize(loss_f);
	case NesterovMomentum:
		return NesterovMomentumOptimizer(step, value(0)).minimize(loss_f);
	case Adagrad:
		return AdagradOptimizer(step, value(2)).minimize(loss_f);
	case RMSProp:
		return RMSPropOptimizer(step, value(0), value(2)).minimize(loss_f);
	case Adam:
		return AdamOptimizer(step, value(0), value(1), value(2)).minimize(loss_f);
	case NesterovAdam:
		return NesterovAdamOptimizer(step, value(0), value(1), value(2)).minimize(loss_f);
	default:
		return nullptr;
	}
}
const Mat nn::CreateMat(int row, int col, int channel, double low, double top)
{
	return mRand(0, int(top - low), row, col, channel, true) + low;
}
const Mat nn::CreateMat(int row, int col, int channel_input, int channel_output, double low, double top)
{
	return mRand(0, int(top - low), row, col, channel_output * channel_input, true) + low;
}
const Mat nn::iconv2d(const Mat & input, const Mat & kern, const Size3 & E_kern_size, ConvInfo conv, bool is_copy_border)
{
	Mat output;
	Size3 area;
	int left, right, top, bottom;
	if (E_kern_size.x == 0 || E_kern_size.y == 0 || E_kern_size.z == 0) {
		area.z = kern.channels() / input.channels();
		if (conv.is_copy_border) {
			area.x = input.rows() * conv.strides.hei;
			area.y = input.cols() * conv.strides.wid;
			output = zeros(input.rows() * conv.strides.hei, input.cols() * conv.strides.wid, area.z);
		}
		else {
			if (conv.anchor == Point(-1, -1)) {
				conv.anchor.x = kern.rows() % 2 ? kern.rows() / 2 : kern.rows() / 2 - 1;
				conv.anchor.y = kern.cols() % 2 ? kern.cols() / 2 : kern.cols() / 2 - 1;
			}
			top = conv.anchor.x;
			bottom = kern.rows() - conv.anchor.x - 1;
			left = conv.anchor.y;
			right = kern.cols() - conv.anchor.y - 1;
			area.x = (input.rows() + top + bottom) * conv.strides.hei;
			area.y = (input.cols() + left + right) * conv.strides.wid;
			output = zeros(area);
		}
		for (int i = 0; i < area.z; i++) {
			Mat sum = zeros(area.x, area.y);
			for (int j = 0; j < input.channels(); j++) {		
				if (!conv.is_copy_border) {
					Mat copy_border = copyMakeBorder(input[j], top, bottom, left, right);
					sum += Filter2D(copy_border, kern[i*input.channels() + j], conv.anchor, conv.strides, is_copy_border);
				}
				else
					sum += Filter2D(input[j], kern[i*input.channels() + j], conv.anchor, conv.strides, is_copy_border);
				
			}
			//output.mChannel(sum / double(input.channels()), i);
			output.mChannel(sum, i);
		}
	}
	else {
		area.z = E_kern_size.z;
		output = zeros(E_kern_size.x, E_kern_size.y, area.z);
		for (int i = 0; i < kern.channels(); i++) {
			for (int j = 0; j < input.channels(); j++) {
				if (conv.is_copy_border) {
					Mat copy_border = copyMakeBorder(input[j], E_kern_size.x / 2, E_kern_size.x / 2, E_kern_size.y / 2, E_kern_size.y / 2);
					output.mChannel(Filter2D(copy_border, kern[i], conv.anchor, conv.strides, is_copy_border), i*input.channels() + j);
				}
				else
					output.mChannel(Filter2D(input[j], kern[i], conv.anchor, conv.strides, is_copy_border), i*input.channels() + j);
			}
		}
	}
	return output;
}const Mat nn::conv2d(const Mat & input, const Mat & kern, const Size & strides, Point anchor, bool is_copy_border)
{
	Mat output;
	if (is_copy_border) {
		output = zeros(input.rows(), input.cols(), kern.channels() / input.channels());
	}
	else {
		int left, right, top, bottom;
		Size3 size = mCalSize(input, kern, anchor, strides, left, right, top, bottom);
		output = zeros(size);
	}
	for (int i = 0; i < kern.channels() / input.channels(); i++) {
		Mat sum = zeros(output.rows(), output.cols());
		for (int j = 0; j < input.channels(); j++)
			sum += Filter2D(input[j], kern[i*input.channels() + j], anchor, strides, is_copy_border);
		output.mChannel(sum, i);
	}
	return output;
}
const Mat nn::upsample(const Mat & input, const Size & ksize, const Mat & markpoint)
{
	if (markpoint.empty())
		return iAveragePool(input, ksize);
	else
		return iMaxPool(input, markpoint, ksize);
}
const Mat nn::MaxPool(const Mat & input, const Size & ksize)
{
	Mat dst = zeros(input.rows() / ksize.hei, input.cols() / ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.cols(); col++) {
				double value = input(0, 0, z);
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						if (value < input(i, j, z)) {
							value = input(i, j, z);
						}
					}
				}
				dst(row, col, z) = value;
			}
	return dst;
}
const Mat nn::MaxPool(const Mat & input, Mat & markpoint, const Size & ksize)
{
	Mat dst = zeros(input.rows() / ksize.hei, input.cols() / ksize.wid, input.channels());
	markpoint = zeros(dst.rows()*dst.cols(), 2, dst.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.cols(); col++) {
				double value = input(0, 0, z);
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						if (value < input(i, j, z)) {
							value = input(i, j, z);
							markpoint(row*dst.cols() + col, 0, z) = i;
							markpoint(row*dst.cols() + col, 1, z) = j;
						}
					}
				}
				dst(row, col, z) = value;
			}
	return dst;
}
const Mat nn::iMaxPool(const Mat & input, const Mat & markpoint, const Size & ksize)
{
	Mat dst = zeros(input.rows() * ksize.hei, input.cols() * ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++) {
		int row = 0;
		for (int i = 0; i < input.rows(); i++)
			for (int j = 0; j < input.cols(); j++) {
				dst((int)markpoint(row, 0, z), (int)markpoint(row, 1, z), z) = input(i, j, z);
				row += 1;
			}
	}
	return dst;
}
const Mat nn::iAveragePool(const Mat & input, const Size & ksize)
{
	Mat dst = zeros(input.rows() * ksize.hei, input.cols() * ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < input.rows(); row++)
			for (int col = 0; col < input.rows(); col++) {
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						dst(i, j, z) = input(row, col, z);
					}
				}
			}
	return dst;
}
const Mat nn::AveragePool(const Mat & input, const Size & ksize)
{
	Mat dst = zeros(input.rows() / ksize.hei, input.cols() / ksize.wid, input.channels());
	for (int z = 0; z < input.channels(); z++)
		for (int row = 0; row < dst.rows(); row++)
			for (int col = 0; col < dst.rows(); col++) {
				double value = 0;
				for (int i = row * ksize.hei; i < row*ksize.hei + ksize.hei; ++i) {
					for (int j = col * ksize.wid; j < col*ksize.wid + ksize.wid; ++j) {
						value += input(i, j, z);
					}
				}
				dst(row, col, z) = value / double(ksize.hei*ksize.wid);
			}
	return dst;
}
const Mat nn::FullConnection(const Mat & input, const Mat & layer, const Mat & bias)
{
	return layer * input + bias;
}
