#ifndef __NETWORK_H__
#define __NETWORK_H__
#include "Mat.h"
#include "Function.h"
#include <vector>
using std::vector;

namespace nn {
/*
===========================    神经网络基类    ============================
*/
	class Optimizer;
	/**
	@brief ConvInfo 卷积参数类
	成员
	Size strides 滑动步长
	Point anchor 像素对应卷积核坐标
	anchor默认为Point(-1,-1), 像素对应卷积核中心
	*/
	class ConvInfo
	{
	public:
		explicit ConvInfo() :strides(), anchor(), is_copy_border(true) {}
		ConvInfo(Size strides, Point anchor, bool is_copy_border)
			: strides(strides), anchor(anchor), is_copy_border(is_copy_border){}
		Size strides;
		Point anchor;
		bool is_copy_border;
	};
	/**
	@brief Activation 激活函数类
	成员
	ActivationFunc activation_f 激活函数
	ActivationFunc activation_df 激活函数导数
	*/
	class Activation
	{
	public:
		ActivationFunc activation_f;
		ActivationFunc activation_df;
	};
	/**
	@brief Loss 损失函数类
	成员
	LossFunc loss_f 损失函数
	LossFunc loss_df 损失函数导数
	*/
	class Loss
	{
	public:
		explicit Loss() :loss_f(nullptr), loss_df(nullptr) {}
		Loss(LossFunc loss_f)
		{
			SetFunc(loss_f, &this->loss_f, &this->loss_df);
		}
		void clean()
		{
			loss_f = nullptr;
			loss_df = nullptr;
		}
		LossFunc loss_f;
		LossFunc loss_df;
	};
	class Net
	{
	public:
		explicit Net();
		Net(Optimizer* op);
		Net(string model_path);
		Net(OptimizerMethod opm, double step = 1e-2, LossFunc loss_f = nullptr);
		~Net();
		/**
		@brief ClearLayer 清除网络参数
		*/
		void ClearLayer();
		/**
		@brief AddActivation 增加激活层
		@param act_f 激活函数
		*/
		void AddActivation(ActivationFunc act_f);
		/**
		@brief AddMaxPool 增加最大值池化层
		@param poolsize 池化层大小
		*/
		void AddMaxPool(Size poolsize);
		/**
		@brief AddMaxPool 增加最大值池化层
		@param pool_row 池化层行数
		@param pool_col 池化层列数
		*/
		void AddMaxPool(int pool_row, int pool_col);
		/**
		@brief AddAveragePool 增加平均值池化层
		@param poolsize 池化层大小
		*/
		void AddAveragePool(Size poolsize);
		/**
		@brief AddMaxPool 增加平均值池化层
		@param pool_row 池化层行数
		@param pool_col 池化层列数
		*/
		void AddAveragePool(int pool_row, int pool_col);
		/**
		@brief AddReshape 增加重置维度层
		不允许改变矩阵大小
		@param re_size 重置二维
		*/
		void AddReshape(Size re_size);
		/**
		@brief AddReshape 增加重置维度层
		不允许改变矩阵大小
		@param re_size 重置三维
		*/
		void AddReshape(Size3 re_size);
		/**
		@brief AddReshape 增加重置维度层
		不允许改变矩阵大小
		@param row 重置行数
		@param col 重置列数
		@param channel 重置通道数
		*/
		void AddReshape(int row, int col, int channel = 1);
		/**
		@brief AddFullConnect 增加全连接层
		权值矩阵的行数必须与偏置矩阵行数相等
		@param full_layer 权值矩阵
		@param full_bias 偏置矩阵
		*/
		void AddFullConnect(const Mat & full_layer, const Mat & full_bias);
		/**
		@brief AddFullConnect 增加全连接层
		权值矩阵的行数必须与偏置矩阵行数相等
		@param layer_row 权值矩阵行数
		@param layer_col 权值矩阵列数
		@param bias_row 偏置矩阵行数
		@param bias_col 偏置矩阵列数
		*/
		void AddFullConnect(int layer_row, int layer_col, int bias_row, int bias_col);
		/**
		@brief AddFullConnect 增加全连接层
		权值矩阵的行数必须与偏置矩阵行数相等
		@param layer_row 权值矩阵行数
		@param layer_col 权值矩阵列数
		@param layer_channel 权值矩阵通道数
		@param bias_row 偏置矩阵行数
		@param bias_col 偏置矩阵列数
		@param bias_channel 偏置矩阵通道数
		*/
		void AddFullConnect(int layer_row, int layer_col, int layer_channel, int bias_row, int bias_col, int bias_channel);
		/**
		@brief AddConv 增加卷积层
		卷积偏置通道数必须与卷积核通道数相等
		@param conv_layer 卷积核
		@param conv_bias 卷积偏置
		@param info 卷积参数
		*/
		void AddConv(const Mat & conv_layer, const Mat & conv_bias, ConvInfo info);
		/**
		@brief AddConv 增加卷积层
		卷积偏置通道数必须与卷积核通道数相等
		@param conv_layer 卷积核
		@param conv_bias 卷积偏置
		@param strides 滑动步长
		@param anchor 像素对应卷积核坐标
		anchor默认为Point(-1,-1), 像素对应卷积核中心
		*/
		void AddConv(const Mat & conv_layer, const Mat & conv_bias, bool is_copy_border = true, Size strides = Size(1, 1), Point anchor = Point(-1, -1));
		/**
		@brief AddConv 增加卷积层
		卷积核大小为kern_row*kern_col*(output_channel-input_channel)
		卷积偏置大小为1*1*(output_channel-input_channel)
		@param kern_row 卷积核行数
		@param kern_col 卷积核列数
		@param input_channel 输入通道数
		@param output_channel 输处通道数
		@param strides 滑动步长
		@param anchor 像素对应卷积核坐标
		anchor默认为Point(-1,-1), 像素对应卷积核中心
		*/
		void AddConv(int kern_row, int kern_col, int input_channel, int output_channel, bool is_copy_border = true, Size strides = Size(1, 1), Point anchor = Point(-1, -1));
		void InitModel(vector<Mat> &layer, vector<LayerType> config, vector<Activation> activation, vector<ConvInfo> convinfo = vector<ConvInfo>(), vector<Size> poolinfo = vector<Size>(), vector<Size3> shape = vector<Size3>());
		double TrainModel(Mat &input, Mat &output);
		double TrainModel(vector<Mat> &input, vector<Mat> &output);
		/**
		@brief Run 前向传播，返回输出结果
		@param input 输入
		*/
		Mat Run(const Mat & input)const;
		/**
		@brief ForwardPropagation 前向传播
		@param input 输入
		@param output 输出
		@param variable 每一层的输出
		*/
		void ForwardPropagation(
			Mat & input, Mat & output,
			const vector<Mat> & layer, vector<Mat>& variable)const;
		/**
		@brief BackPropagation 反向传播
		@param input 输入
		@param output 输出
		@param d_layer 输出梯度值
		*/
		double BackPropagation(
			Mat & input, Mat & output,
			vector<Mat> & d_layer);
		/**
		@brief FutureJacobi 预测雅可比矩阵
		模型参数先进行迭代后计算梯度矩阵
		@param input 输入
		@param output 输出
		@param d_layer 输出梯度矩阵
		*/
		double FutureJacobi(
			Mat & input, Mat & output,
			vector<Mat> & d_layer)const;
		/**
		@brief FutureJacobi 雅可比矩阵
		计算梯度矩阵
		@param input 输入
		@param output 输出
		@param d_layer 输出梯度矩阵
		*/
		double Jacobi(
			Mat & input, Mat & output,
			vector<Mat> & d_layer)const;
		void JacobiMat(vector<Mat> &d_layer, const vector<Mat> &x,
			const Mat &y, const Mat &output)const;

		/**
		@brief LayerNum 返回权值vector长度
		*/
		size_t LayerNum()const;
		/**
		@brief Run 返回运行结果对错
		@param input 输入
		@param output 标签
		*/
		bool Run(Mat &input, Mat &output)const;
		/**
		@brief Accuracy 返回准确率
		@param input 输入
		@param output 标签
		*/
		double Accuracy(vector<Mat> &input, vector<Mat> &output)const;
		/**
		@brief InitMethod 注册优化方法 
		@param optrimizer 优化器
		*/
		void InitMethod(Optimizer* optrimizer);
		void ReadModel(string model_path);
		void SaveModel(string save_path = "./model")const;

	protected:
		void AddConv(bool is_copy_border, Size strides, Point anchor);
	private:
		//优化器
		Optimizer* op;
		//权值矩阵
		vector<Mat> layer;
		//重置维度参数
		vector<Size3> shape;
		//池化参数
		vector<Size> poolinfo;
		//网络层的类型
		vector<LayerType> config;
		//卷积参数
		vector<ConvInfo> convinfo;
		//激活函数
		vector<Activation> activation;
	};

	/**
	@brief CreateMat 创建随机矩阵
	大小为row*col*channel, 元素范围[low, top]
	@param row 矩阵行数
	@param col 矩阵列数
	@param channel 矩阵通道数
	@param low 下限
	@param top 上限
	*/
	Mat CreateMat(int row, int col, int channel = 1, double low = -0.5, double top = 0.5);
	/**
	@brief CreateMat 创建随机矩阵
	大小为row*col*(channel_output - channel_input), 元素范围[low, top]
	@param row 矩阵行数
	@param col 矩阵列数
	@param channel_input 输入矩阵通道数
	@param channel_output 输出矩阵通道数
	@param low 下限
	@param top 上限
	*/
	Mat CreateMat(int row, int col, int channel_input, int channel_output, double low = -0.5, double top = 0.5);
	const Mat iconv2d(const Mat& input, const Mat& kern, const Size3 & E_kern_size, const Size& strides, Point anchor, bool is_copy_border = true);
	const Mat conv2d(const Mat& input, const Mat& kern, const Size& strides, Point anchor, bool is_copy_border = true);
	const Mat upsample(const Mat & input, const Size & ksize, const Mat & markpoint = Mat());
	const Mat MaxPool(const Mat & input, const Size & ksize);
	const Mat MaxPool(const Mat & input, Mat & markpoint, const Size & ksize);
	const Mat iMaxPool(const Mat & input, const Mat & markpoint, const Size & ksize);
	const Mat iAveragePool(const Mat& input, const Size & ksize);
	const Mat AveragePool(const Mat& input, const Size & ksize);
	const Mat FullConnection(const Mat & input, const Mat & layer, const Mat & bias);
}

#endif //__NETWORK_H__