#ifndef __TRAIN_H__
#define __TRAIN_H__

#include "Net.h"
#include <vector>
#include <string>
using std::vector;
using std::string;


namespace nn {
	/**
	@brief Optimizer�Ż�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����
	*/
	class Optimizer
	{
	public:
		explicit Optimizer();
		Optimizer(double step);
		~Optimizer();
		/**
		@brief ��ʼ���Ż�������
		@param size �Ż�����ߴ�
		*/
		virtual void init(vector<Size3>& size) = 0;
		/**
		@brief �����Ż�������1��
		@param x ��������
		@param y �������
		@param da �仯��
		@param size
		@param size
		*/
		virtual int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error) = 0;
		void RegisterNet(Net *net);
		bool Enable(const Mat &x, const Mat&y, vector<Mat>& a)const;
		void RegisterMethod(OptimizerMethod method);
		OptimizerMethod Method()const;
		Loss loss;
	protected:
		//����ʵ��
		Net * net;
		//ѧϰ��
		double step;
		//�̳л��������
		OptimizerMethod method;
	};
	/**
	@brief Method����ĺ�������
	ע��ģ�͵���ʧ����loss
	���ṩ�Ż�����
	*/
	class Method :public Optimizer
	{
	public:
		explicit Method();
		Method(double step);
		void init(vector<Size3>& size) {}
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		Optimizer* minimize(LossFunc loss_ = nullptr)const;
	};
	/**
	@brief GradientDescentOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����GradientDescent
	a = a - step * df(a, x)
	*/
	class GradientDescentOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		*/
		explicit GradientDescentOptimizer(double step = 1e-2);
		GradientDescentOptimizer(vector<double>& value);
		void init(vector<Size3>& size) {}
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;
		Mat data(vector<string> &value_name)const;
	};
	/**
	@brief MomentumOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����MomentumGradientDescent
	ma = momentum*ma + step * df(a, x)
	a = a - ma
	*/
	class MomentumOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param gama ����ϵ��
		*/
		explicit MomentumOptimizer(double step = 1e-2, double gama = 0.9);
		MomentumOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;

		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> ma;
		double momentum;
	};
	/**
	@brief NesterovMomentumOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����NesterovMomentumGradientDescent
	ma = momentum*ma + step * df(a - momentum*ma, x)
	a = a - ma
	*/
	class NesterovMomentumOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param gama ����ϵ��
		*/
		explicit NesterovMomentumOptimizer(double step = 1e-2, double gama = 0.9);
		NesterovMomentumOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const; 
		
		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> ma;
		double momentum;
	};
	/**
	@brief AdagradOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����AdagradGradientDescent
	alpha = alpha + df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	class AdagradOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param epsilon ƫ��(�����ĸ=0)
		*/
		explicit AdagradOptimizer(double step = 1e-2, double epsilon = 1e-7); 
		AdagradOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;
		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> alpha;
		double epsilon;
	};
	/**
	@brief RMSPropOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����RMSPropGradientDescent
	alpha = beta*alpha + (1 - beta)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	class RMSPropOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param decay ����Ӧ����
		@param epsilon ƫ��(�����ĸ=0)
		*/
		explicit RMSPropOptimizer(double step = 1e-2, double decay = 0.9, double epsilon = 1e-7);
		RMSPropOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;
		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> alpha;
		double decay;
		double epsilon;
	};
	/**
	@brief AdamOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����AdamGradientDescent
	ma = beta1*ma + (1 - beta1)*df(a, x)
	alpha = beta2*alpha + (1 - beta2)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	class AdamOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param beta1 ����ϵ��
		@param beta2 ����Ӧϵ��
		@param epsilon ƫ��(�����ĸ=0)
		*/
		explicit AdamOptimizer(double step = 1e-2, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-7);
		AdamOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;
		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> ma;
		vector<Mat> alpha;
		double epsilon;
		double beta1;
		double beta2;
	};
	/**
	@brief NesterovAdamOptimizer����ĺ�������
	ע��ģ�͵���ʧ����loss
	�ṩ�Ż�����NesterovAdamGradientDescent
	ma = beta1*ma + (1 - beta1)*df(a - step/sqrt(alpha + epsilon)*ma, x)
	alpha = beta2*alpha + (1 - beta2)*df(a - step/sqrt(alpha + epsilon)*ma, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	class NesterovAdamOptimizer :public Optimizer
	{
	public:
		/**
		@brief �Ż������캯��
		@param step ѧϰ��
		@param beta1 ����ϵ��
		@param beta2 ����Ӧϵ��
		@param epsilon ƫ��(�����ĸ=0)
		*/
		explicit NesterovAdamOptimizer(double step = 1e-2, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-7);
		NesterovAdamOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief ����ģ�ͺ���������ע����Ż���Optimizer
		@param loss_ ��ʧ����
		@param output_ �������
		@param activation_ �����
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;
		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> ma;
		vector<Mat> alpha;
		double epsilon;
		double beta1;
		double beta2;
	};
	typedef Optimizer* OptimizerP;
}

#endif // __TRAIN_H__