#include <iostream>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <iomanip>
#include <io.h>
#include <direct.h>
#include "include.h"

using namespace nn;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

//读取训练集，输入保存到input向量，标签保存到output向量中
void read_data(vector<Mat> &input, vector<Mat> &label)
{
	ifstream file("data.txt");
	if (!file.is_open())return;
	//row训练集个数, col单个训练集的长度, count标签长度
	int row, col, count;
	file >> row;
	file >> col;
	file >> count;
	vector<Mat>(row).swap(input);
	vector<Mat>(row).swap(label);
	for (int i = 0; i < row; i++) {
		int index;
		label[i] = zeros(count, 1);
		input[i] = zeros(col, 1);
		file >> index;
		label[i](index) = 1;
		for (int j = 0; j < col; j++) {
			file >> index;
			input[i](j) = index;
		}
	}
	file.close();
}
//读取图像并归一化
const Mat ReadImageData(const Image &input)
{
	Image dst = 255 - input;//黑白转换
	Mat mat = Image2Mat(dst);//图像转为矩阵
	mat /= Max(mat);//归一化
	mat.reshape(16 * 16, 1, 1);
	return mat;
}
//测试网络
void testNet(string file_path, string model_path, string finish_save_path)
{
	vector<string> files;
	getFiles("train_data", files);//保存所有文件路径
	if (files.empty()) {
		printf("file is empty!\n");
		return;
	}
	string word = "ABCDEFGHJKLMNPRSTUVWXYZ";//所有结果
	Net nn(model_path);//读取模型构建网络
	vector<char> data_c(files.size());
	vector<int> data_i(files.size());
	for (size_t i = 0; i < files.size(); ++i) {
		Image image = Imread(files[i].c_str(), true);//用灰度读取图像
		Mat output = nn.Run(ReadImageData(image));//运行网络输出
		int adr = output.maxAt();//取最大值索引
		data_c[i] = word[adr];
		data_i[i] = adr;
		cout << word[adr] << ' ' << adr << '\t';
		cout << files[i] << endl;
	}
	if (finish_save_path.empty()) return;
	//保存数据
	ofstream out(finish_save_path);
	if (out.is_open()) {
		for (size_t i = 0; i < files.size(); i++)
			out << data_c[i] << ',' << data_i[i] << ',' << files[i] << endl;
		out.close();
	}

}
//构建网络
void CreateNN(Net & nn)
{
	nn.AddFullConnect(CreateMat(16 * 16, 50), CreateMat(50, 1));//增加全连接 16*16 -> 100
	nn.AddActivation(Sigmoid);//增加激活层Sigmoid
	//nn.AddDropout(0.3);//增加dropout层, 丢弃率为0.3
	nn.AddFullConnect(CreateMat(50, 50), CreateMat(50, 1));//增加全连接 100 -> 50
	nn.AddActivation(Sigmoid);//增加激活层Sigmoid
	nn.AddFullConnect(CreateMat(50, 23), CreateMat(23, 1));//增加全连接 50 -> 23
	nn.AddActivation(Softmax);//增加激活层Softmax
	nn.InitMethod(CreateOptimizer(GradientDescent, 5e-1, Quadratic));//注册优化器Momentum, 学习率为0.02, 损失函数为平方差
}
void CreateCNN(Net & cnn)
{
	cnn.AddReshape(16, 16);//增加重置维度层将(16*16, 1, 1)->(16, 16, 1)
	cnn.AddConv(CreateMat(5, 5, 1, 10), CreateMat(1, 1, 10));//增加卷积层(卷积核为5*5*10
	cnn.AddActivation(ReLU);//增加激活层ReLU
	cnn.AddMaxPool(2, 2);//增加最大值池化层(池化大小为2,2 <! 长宽缩小一半
	cnn.AddConv(CreateMat(3, 3, 10, 30), CreateMat(1, 1, 30));//增加卷积层(卷积核为3*3*3
	cnn.AddActivation(ReLU);//增加激活层ReLU
	cnn.AddMaxPool(2, 2);//增加最大值池化层(池化大小为2,2 <! 长宽缩小一半
	cnn.AddReshape(4 * 4 * 30, 1);//增加重置维度层将(4, 4, 30)->(4*4*30, 1, 1)
	cnn.AddFullConnect(CreateMat(4 * 4 * 30, 256), CreateMat(256, 1));//增加全连接层4*4*30 -> 256
	cnn.AddActivation(ReLU);//增加激活层ReLU
	cnn.AddFullConnect(CreateMat(256, 23), CreateMat(23, 1));//增加全连接层256 ->23
	cnn.AddActivation(Softmax);//增加激活层Softmax
	cnn.InitMethod(CreateOptimizer(Adam, 1e-2, CrossEntropy));//注册优化器Adam, 学习率为0.02, 损失函数为交叉熵
}
void NetTrain(vector<Mat> &input, vector<Mat> &label, Net &net, int n, int m, int times)
{
	if (m == input.size()) {
		double acc = 0;
		for (int i = 0; i <= n; ++i) {
			acc = net.TrainModel(input, label);
			cout << "迭代次数：" << i << "\t\t总准确率：" << acc * 100 << "%" << endl;
			if (acc == 1)break;
		}
	}
	else {
		//生成抽样索引集
		vector<int> randomVec;
		for (int i = 0; i < (int)input.size(); ++i)
			randomVec.push_back(i);
		//正确率
		double acc = 0;
		//误差
		double error = 0;
		double d_error = 0;
		//正确率
		double mean_acc = 0;
		//误差
		double mean_error = 0;
		//百分百正确率出现的次数
		int count = 0;
		bool calall = false;
		for (int i = 0; i <= n; ++i) {
			//乱序
			random_shuffle(randomVec.begin(), randomVec.end());
			//抽样训练集输入
			vector<Mat> in(m);
			//抽样训练集标签
			vector<Mat> out(m);
			for (int j = 0; j < m; ++j) {
				in[j] = input[randomVec[j]];
				out[j] = label[randomVec[j]];
			}
			//训练模型
			acc = net.TrainModel(in, out, &error);
			if (i%times == 0) {				
				printf("迭代次数：%d \t样本正确率：%0.2lf%% \t误差：%0.4e\n", i, acc * 100, error);
			}
		}
	}
}
void train(vector<Mat> & input, vector<Mat> & label, Net &net, int n, int m, int times, string save_path)
{
	NetTrain(input, label, net, n, m, times);
	printf("总正确率：%0.4lf%%\n", net.Accuracy(input, label) * 100);
	net.SaveModel(save_path);
}
int main()
{
	clock_t start, end;
	if (0 != _access("data.txt", 0))
	{
		system("python totxt.py");
	}
	vector<Mat> input;
	vector<Mat> output;
	read_data(input, output);
	start = clock();
	Srandom();
	string save_nn_path = "./model/nn_model.conf";
	string save_cnn_path = "./model/cnn_model.conf";
	Net net;
	//全连接网络
	CreateNN(net);
	train(input, output, net, 1000, 50, 100, save_nn_path);
	//卷积网络
	/*CreateCNN(net);
	train(input, output, net, 300, 50, 1, save_cnn_path);*/
	//全连接测试网络
	testNet("./train_data", save_nn_path, "nn_finish.csv");
	//卷积测试网络
	//testNet("./train_data", save_cnn_path, "cnn_finish.csv");
	end = clock();
	printf("\ntime=%fs\n", (double)(end - start) / CLK_TCK);
	system("pause");
	return 0;
}