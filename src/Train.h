#ifndef __TRAIN_H__
#define __TRAIN_H__

#include "Net.h"
#include <vector>
#include <string>
using std::vector;
using std::string;


namespace nn {
	/**
	@brief Optimizer优化器基类
	注册模型的损失函数loss
	提供优化方法
	*/
	class Optimizer
	{
	public:
		explicit Optimizer();
		Optimizer(double step);
		~Optimizer();
		/**
		@brief 初始化优化器参数
		@param size 优化矩阵尺寸
		*/
		virtual void init(vector<Size3>& size) = 0;
		/**
		@brief 运行优化器迭代1次
		@param x 网络输入
		@param y 网络输出
		@param da 变化量
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
		//网络实体
		Net * net;
		//学习率
		double step;
		//继承基类的类型
		OptimizerMethod method;
	};
	/**
	@brief Method网络的函数配置
	注册模型的损失函数loss
	不提供优化方法
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
	@brief GradientDescentOptimizer网络的函数配置
	注册模型的损失函数loss
	提供优化方法GradientDescent
	a = a - step * df(a, x)
	*/
	class GradientDescentOptimizer :public Optimizer
	{
	public:
		/**
		@brief 优化器构造函数
		@param step 学习率
		*/
		explicit GradientDescentOptimizer(double step = 1e-2);
		GradientDescentOptimizer(vector<double>& value);
		void init(vector<Size3>& size) {}
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief 配置模型函数，返回注册的优化器Optimizer
		@param loss_ 损失函数
		@param output_ 输出函数
		@param activation_ 激活函数
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;
		Mat data(vector<string> &value_name)const;
	};
	/**
	@brief MomentumOptimizer网络的函数配置
	注册模型的损失函数loss
	提供优化方法MomentumGradientDescent
	ma = momentum*ma + step * df(a, x)
	a = a - ma
	*/
	class MomentumOptimizer :public Optimizer
	{
	public:
		/**
		@brief 优化器构造函数
		@param step 学习率
		@param gama 动量系数
		*/
		explicit MomentumOptimizer(double step = 1e-2, double gama = 0.9);
		MomentumOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief 配置模型函数，返回注册的优化器Optimizer
		@param loss_ 损失函数
		@param output_ 输出函数
		@param activation_ 激活函数
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;

		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> ma;
		double momentum;
	};
	/**
	@brief NesterovMomentumOptimizer网络的函数配置
	注册模型的损失函数loss
	提供优化方法NesterovMomentumGradientDescent
	ma = momentum*ma + step * df(a - momentum*ma, x)
	a = a - ma
	*/
	class NesterovMomentumOptimizer :public Optimizer
	{
	public:
		/**
		@brief 优化器构造函数
		@param step 学习率
		@param gama 动量系数
		*/
		explicit NesterovMomentumOptimizer(double step = 1e-2, double gama = 0.9);
		NesterovMomentumOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief 配置模型函数，返回注册的优化器Optimizer
		@param loss_ 损失函数
		@param output_ 输出函数
		@param activation_ 激活函数
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const; 
		
		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> ma;
		double momentum;
	};
	/**
	@brief AdagradOptimizer网络的函数配置
	注册模型的损失函数loss
	提供优化方法AdagradGradientDescent
	alpha = alpha + df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	class AdagradOptimizer :public Optimizer
	{
	public:
		/**
		@brief 优化器构造函数
		@param step 学习率
		@param epsilon 偏置(避免分母=0)
		*/
		explicit AdagradOptimizer(double step = 1e-2, double epsilon = 1e-7); 
		AdagradOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief 配置模型函数，返回注册的优化器Optimizer
		@param loss_ 损失函数
		@param output_ 输出函数
		@param activation_ 激活函数
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;
		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> alpha;
		double epsilon;
	};
	/**
	@brief RMSPropOptimizer网络的函数配置
	注册模型的损失函数loss
	提供优化方法RMSPropGradientDescent
	alpha = beta*alpha + (1 - beta)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*df(a, x)
	*/
	class RMSPropOptimizer :public Optimizer
	{
	public:
		/**
		@brief 优化器构造函数
		@param step 学习率
		@param decay 自适应参数
		@param epsilon 偏置(避免分母=0)
		*/
		explicit RMSPropOptimizer(double step = 1e-2, double decay = 0.9, double epsilon = 1e-7);
		RMSPropOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief 配置模型函数，返回注册的优化器Optimizer
		@param loss_ 损失函数
		@param output_ 输出函数
		@param activation_ 激活函数
		*/
		Optimizer* minimize(LossFunc loss_ = nullptr)const;
		Mat data(vector<string> &value_name)const;
	private:
		vector<Mat> alpha;
		double decay;
		double epsilon;
	};
	/**
	@brief AdamOptimizer网络的函数配置
	注册模型的损失函数loss
	提供优化方法AdamGradientDescent
	ma = beta1*ma + (1 - beta1)*df(a, x)
	alpha = beta2*alpha + (1 - beta2)*df(a, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	class AdamOptimizer :public Optimizer
	{
	public:
		/**
		@brief 优化器构造函数
		@param step 学习率
		@param beta1 动量系数
		@param beta2 自适应系数
		@param epsilon 偏置(避免分母=0)
		*/
		explicit AdamOptimizer(double step = 1e-2, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-7);
		AdamOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief 配置模型函数，返回注册的优化器Optimizer
		@param loss_ 损失函数
		@param output_ 输出函数
		@param activation_ 激活函数
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
	@brief NesterovAdamOptimizer网络的函数配置
	注册模型的损失函数loss
	提供优化方法NesterovAdamGradientDescent
	ma = beta1*ma + (1 - beta1)*df(a - step/sqrt(alpha + epsilon)*ma, x)
	alpha = beta2*alpha + (1 - beta2)*df(a - step/sqrt(alpha + epsilon)*ma, x)^2
	a = a - step/sqrt(alpha + epsilon)*ma
	*/
	class NesterovAdamOptimizer :public Optimizer
	{
	public:
		/**
		@brief 优化器构造函数
		@param step 学习率
		@param beta1 动量系数
		@param beta2 自适应系数
		@param epsilon 偏置(避免分母=0)
		*/
		explicit NesterovAdamOptimizer(double step = 1e-2, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-7);
		NesterovAdamOptimizer(vector<double>& value);
		void init(vector<Size3>& size);
		int Run(vector<Mat> &dlayer, const Mat &x, const Mat &y, vector<Mat> &a, double &error);
		/**
		@brief 配置模型函数，返回注册的优化器Optimizer
		@param loss_ 损失函数
		@param output_ 输出函数
		@param activation_ 激活函数
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