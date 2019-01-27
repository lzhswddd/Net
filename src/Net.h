#ifndef __NETWORK_H__
#define __NETWORK_H__
#include "Mat.h"
#include "Function.h"
#include <vector>
using std::vector;

namespace nn {
/*
===========================    ���������    ============================
*/
	class Optimizer;
	/**
	@brief ConvInfo ���������
	��Ա
	Size strides ��������
	Point anchor ���ض�Ӧ���������
	anchorĬ��ΪPoint(-1,-1), ���ض�Ӧ���������
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
	@brief Activation �������
	��Ա
	ActivationFunc activation_f �����
	ActivationFunc activation_df ���������
	*/
	class Activation
	{
	public:
		ActivationFunc activation_f;
		ActivationFunc activation_df;
	};
	/**
	@brief Loss ��ʧ������
	��Ա
	LossFunc loss_f ��ʧ����
	LossFunc loss_df ��ʧ��������
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
	/**
	@brief NetConfig ���������
	��ԱΪ���е��������
	Net���ṩ��InitModel��������ע��
	*/
	struct NetConfig
	{
		//�Ż���
		Optimizer* op;
		//Ȩֵ����
		vector<Mat> layer = vector<Mat>();
		//Dropout�㶪������
		vector<double> dropout = vector<double>();
		//����ά�Ȳ���
		vector<Size3> shape = vector<Size3>();
		//�ػ�����
		vector<Size> poolinfo = vector<Size>();
		//����������
		vector<LayerType> config = vector<LayerType>();
		//�������
		vector<ConvInfo> convinfo = vector<ConvInfo>();
		//�����
		vector<Activation> activation = vector<Activation>();
	};
	/**
	@brief Net ������
	�ܹ��������磬����ע���Ż���������½���ѵ��
	ӵ��ȫ���Ӳ㡢����㡢����㡢�ػ��㡢����ά�Ȳ㡢dropout��
	��Ҫע���Ӧ�Ĳ��������ܹ�������
	�ṩ�˶�Ӧ�ķ��������ڿ��ٹ�������
	InitMethod		ע���Ż���
	AddDropout		���dropout��
	AddActivation	��Ӽ����
	AddMaxPool		������ֵ�ػ���
	AddAveragePool	���ƽ��ֵ�ػ���
	AddReshape		�������ά�Ȳ�
	AddFullConnect	���ȫ���Ӳ�
	AddConv			��Ӿ����
	*/
	class Net
	{
	public:
		explicit Net();
		Net(Optimizer* op);
		Net(string model_path);
		Net(OptimizerMethod opm, double step = 1e-2, LossFunc loss_f = nullptr);
		~Net();
		/**
		@brief ClearLayer ����������
		*/
		void ClearLayer();
		/**
		@brief AddDropout ����dropout��
		@param dropout_prob ��������
		*/
		void AddDropout(double dropout_prob);
		/**
		@brief AddActivation ���Ӽ����
		@param act_f �����
		*/
		void AddActivation(ActivationFunc act_f);
		/**
		@brief AddMaxPool �������ֵ�ػ���
		@param poolsize �ػ����С
		*/
		void AddMaxPool(Size poolsize);
		/**
		@brief AddMaxPool �������ֵ�ػ���
		@param pool_row �ػ�������
		@param pool_col �ػ�������
		*/
		void AddMaxPool(int pool_row, int pool_col);
		/**
		@brief AddAveragePool ����ƽ��ֵ�ػ���
		@param poolsize �ػ����С
		*/
		void AddAveragePool(Size poolsize);
		/**
		@brief AddMaxPool ����ƽ��ֵ�ػ���
		@param pool_row �ػ�������
		@param pool_col �ػ�������
		*/
		void AddAveragePool(int pool_row, int pool_col);
		/**
		@brief AddReshape ��������ά�Ȳ�
		������ı�����С
		@param re_size ���ö�ά
		*/
		void AddReshape(Size re_size);
		/**
		@brief AddReshape ��������ά�Ȳ�
		������ı�����С
		@param re_size ������ά
		*/
		void AddReshape(Size3 re_size);
		/**
		@brief AddReshape ��������ά�Ȳ�
		������ı�����С
		@param row ��������
		@param col ��������
		@param channel ����ͨ����
		*/
		void AddReshape(int row, int col, int channel = 1);
		/**
		@brief AddFullConnect ����ȫ���Ӳ�
		Ȩֵ���������������ƫ�þ����������
		@param full_layer Ȩֵ����
		@param full_bias ƫ�þ���
		*/
		void AddFullConnect(const Mat & full_layer, const Mat & full_bias);
		/**
		@brief AddFullConnect ����ȫ���Ӳ�
		Ȩֵ���������������ƫ�þ����������
		@param layer_row Ȩֵ��������
		@param layer_col Ȩֵ��������
		@param bias_row ƫ�þ�������
		@param bias_col ƫ�þ�������
		*/
		void AddFullConnect(int layer_row, int layer_col, int bias_row, int bias_col);
		/**
		@brief AddFullConnect ����ȫ���Ӳ�
		Ȩֵ���������������ƫ�þ����������
		@param layer_row Ȩֵ��������
		@param layer_col Ȩֵ��������
		@param layer_channel Ȩֵ����ͨ����
		@param bias_row ƫ�þ�������
		@param bias_col ƫ�þ�������
		@param bias_channel ƫ�þ���ͨ����
		*/
		void AddFullConnect(int layer_row, int layer_col, int layer_channel, int bias_row, int bias_col, int bias_channel);
		/**
		@brief AddConv ���Ӿ����
		���ƫ��ͨ��������������ͨ�������
		@param conv_layer �����
		@param conv_bias ���ƫ��
		@param info �������
		*/
		void AddConv(const Mat & conv_layer, const Mat & conv_bias, ConvInfo info);
		/**
		@brief AddConv ���Ӿ����
		���ƫ��ͨ��������������ͨ�������
		@param conv_layer �����
		@param conv_bias ���ƫ��
		@param is_copy_border ���ʱ�Ƿ�����߽�
		@param strides ��������
		@param anchor ���ض�Ӧ���������
		anchorĬ��ΪPoint(-1,-1), ���ض�Ӧ���������
		*/
		void AddConv(const Mat & conv_layer, const Mat & conv_bias, bool is_copy_border = true, Size strides = Size(1, 1), Point anchor = Point(-1, -1));
		/**
		@brief AddConv ���Ӿ����
		����˴�СΪkern_row*kern_col*(output_channel-input_channel)
		���ƫ�ô�СΪ1*1*(output_channel-input_channel)
		@param kern_row ���������
		@param kern_col ���������
		@param input_channel ����ͨ����
		@param output_channel �䴦ͨ����
		@param is_copy_border ���ʱ�Ƿ�����߽�
		@param strides ��������
		@param anchor ���ض�Ӧ���������
		anchorĬ��ΪPoint(-1,-1), ���ض�Ӧ���������
		*/
		void AddConv(int kern_row, int kern_col, int input_channel, int output_channel, bool is_copy_border = true, Size strides = Size(1, 1), Point anchor = Point(-1, -1));

		void InitModel(struct NetConfig & config_);
		void SetLayer(vector<Mat> & layer);
		void SetLayerType(vector<LayerType> & config);
		void SetConvinfo(vector<ConvInfo> & convinfo);
		void SetPoolinfo(vector<Size> & poolinfo);
		void SetReshape(vector<Size3> & shape);
		void SetActivation(vector<Activation> & activation);
		void SetDropout(vector<double> & dropout);

		/**
		@brief TrainModel ������ѵ��ģ��
		return ����1��ȷ, 0����, -1�쳣
		@param input ����
		@param label ��ǩ
		@param error ��ʧ�������
		*/
		int TrainModel(const Mat &input, const Mat &label, double *error = nullptr);
		/**
		@brief TrainModel ������ѵ��ģ��
		return ������ȷ��, -1�쳣
		@param input ����
		@param label ��ǩ
		@param error ��ʧ�������
		*/
		double TrainModel(const vector<Mat> &input, const vector<Mat> &label, double *error = nullptr);
		/**
		@brief Run ǰ�򴫲�������������
		@param input ����
		*/
		Mat Run(const Mat & input)const;
		/**
		@brief ForwardPropagation ǰ�򴫲�
		@param input ����
		@param output ���
		@param variable ÿһ������
		*/
		void ForwardPropagation(
			const Mat & input, Mat & output,
			const vector<Mat> & layer, vector<Mat>& variable)const;
		/**
		@brief BackPropagation ���򴫲�
		@param input ����
		@param output ���
		@param d_layer ����ݶ�ֵ
		*/
		int BackPropagation(
			const Mat & input, const Mat & output,
			vector<Mat> & d_layer, double & error);
		/**
		@brief FutureJacobi Ԥ���ſɱȾ���
		ģ�Ͳ����Ƚ��е���������ݶȾ���
		@param input ����
		@param output ���
		@param d_layer ����ݶȾ���
		*/
		int FutureJacobi(
			const Mat & input, const Mat & output,
			vector<Mat> & d_layer, double & error)const;
		/**
		@brief FutureJacobi �ſɱȾ���
		�����ݶȾ���
		@param input ����
		@param output ���
		@param d_layer ����ݶȾ���
		*/
		int Jacobi(
			const Mat & input, const Mat & output,
			vector<Mat> & d_layer, double & error)const;
		void JacobiMat(vector<Mat> &d_layer, const vector<Mat> &x,
			const Mat &y, const Mat &output)const;

		/**
		@brief LayerNum ����Ȩֵvector����
		*/
		size_t LayerNum()const;
		/**
		@brief Run �������н���Դ�
		@param input ����
		@param output ��ǩ
		*/
		bool Run(const Mat &input, const Mat &output)const;
		/**
		@brief Accuracy ����׼ȷ��
		@param input ����
		@param output ��ǩ
		*/
		double Accuracy(const vector<Mat> &input, const vector<Mat> &output)const;
		/**
		@brief InitMethod ע���Ż����� 
		@param optrimizer �Ż���
		*/
		void InitMethod(Optimizer* optrimizer);
		void ReadModel(string model_path);
		void SaveModel(string save_path = "./model")const;

	protected:
		void AddConv(bool is_copy_border, Size strides, Point anchor);
	private:
		//�Ż���
		Optimizer* op;
		//Ȩֵ����
		vector<Mat> layer;
		//Dropout�㶪������
		vector<double> dropout;
		//����ά�Ȳ���
		vector<Size3> shape;
		//�ػ�����
		vector<Size> poolinfo;
		//����������
		vector<LayerType> config;
		//�������
		vector<ConvInfo> convinfo;
		//�����
		vector<Activation> activation;
	};
	/**
	@brief CreateOptimizer �����Ż���
	�����Ż���
	@param opm �Ż�������
	@param step ѧϰ��
	@param loss_f ��ʧ����
	@param value ����
	*/
	Optimizer* CreateOptimizer(OptimizerMethod opm, double step = 1e-2, LossFunc loss_f = nullptr, const Mat &value = (Mat_(3, 1) << 9e-1, 999e-3, 1e-7));
	/**
	@brief CreateMat �����������
	��СΪrow*col*channel, Ԫ�ط�Χ[low, top]
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	@param low ����
	@param top ����
	*/
	const Mat CreateMat(int row, int col, int channel = 1, double low = -0.5, double top = 0.5);
	/**
	@brief CreateMat �����������
	��СΪrow*col*(channel_output - channel_input), Ԫ�ط�Χ[low, top]
	@param row ��������
	@param col ��������
	@param channel_input �������ͨ����
	@param channel_output �������ͨ����
	@param low ����
	@param top ����
	*/
	const Mat CreateMat(int row, int col, int channel_input, int channel_output, double low = -0.5, double top = 0.5);
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