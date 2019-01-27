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

//��ȡѵ���������뱣�浽input��������ǩ���浽output������
void read_data(vector<Mat> &input, vector<Mat> &label)
{
	ifstream file("data.txt");
	if (!file.is_open())return;
	//rowѵ��������, col����ѵ�����ĳ���, count��ǩ����
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
//��ȡͼ�񲢹�һ��
const Mat ReadImageData(const Image &input)
{
	Image dst = 255 - input;//�ڰ�ת��
	Mat mat = Image2Mat(dst);//ͼ��תΪ����
	mat /= Max(mat);//��һ��
	mat.reshape(16 * 16, 1, 1);
	return mat;
}
//��������
void testNet(string file_path, string model_path, string finish_save_path)
{
	vector<string> files;
	getFiles("train_data", files);//���������ļ�·��
	if (files.empty()) {
		printf("file is empty!\n");
		return;
	}
	string word = "ABCDEFGHJKLMNPRSTUVWXYZ";//���н��
	Net nn(model_path);//��ȡģ�͹�������
	vector<char> data_c(files.size());
	vector<int> data_i(files.size());
	for (size_t i = 0; i < files.size(); ++i) {
		Image image = Imread(files[i].c_str(), true);//�ûҶȶ�ȡͼ��
		Mat output = nn.Run(ReadImageData(image));//�����������
		int adr = output.maxAt();//ȡ���ֵ����
		data_c[i] = word[adr];
		data_i[i] = adr;
		cout << word[adr] << ' ' << adr << '\t';
		cout << files[i] << endl;
	}
	if (finish_save_path.empty()) return;
	//��������
	ofstream out(finish_save_path);
	if (out.is_open()) {
		for (size_t i = 0; i < files.size(); i++)
			out << data_c[i] << ',' << data_i[i] << ',' << files[i] << endl;
		out.close();
	}

}
//��������
void CreateNN(Net & nn)
{
	nn.AddFullConnect(CreateMat(16 * 16, 50), CreateMat(50, 1));//����ȫ���� 16*16 -> 100
	nn.AddActivation(Sigmoid);//���Ӽ����Sigmoid
	//nn.AddDropout(0.3);//����dropout��, ������Ϊ0.3
	nn.AddFullConnect(CreateMat(50, 50), CreateMat(50, 1));//����ȫ���� 100 -> 50
	nn.AddActivation(Sigmoid);//���Ӽ����Sigmoid
	nn.AddFullConnect(CreateMat(50, 23), CreateMat(23, 1));//����ȫ���� 50 -> 23
	nn.AddActivation(Softmax);//���Ӽ����Softmax
	nn.InitMethod(CreateOptimizer(GradientDescent, 5e-1, Quadratic));//ע���Ż���Momentum, ѧϰ��Ϊ0.02, ��ʧ����Ϊƽ����
}
void CreateCNN(Net & cnn)
{
	cnn.AddReshape(16, 16);//��������ά�Ȳ㽫(16*16, 1, 1)->(16, 16, 1)
	cnn.AddConv(CreateMat(5, 5, 1, 10), CreateMat(1, 1, 10));//���Ӿ����(�����Ϊ5*5*10
	cnn.AddActivation(ReLU);//���Ӽ����ReLU
	cnn.AddMaxPool(2, 2);//�������ֵ�ػ���(�ػ���СΪ2,2 <! ������Сһ��
	cnn.AddConv(CreateMat(3, 3, 10, 30), CreateMat(1, 1, 30));//���Ӿ����(�����Ϊ3*3*3
	cnn.AddActivation(ReLU);//���Ӽ����ReLU
	cnn.AddMaxPool(2, 2);//�������ֵ�ػ���(�ػ���СΪ2,2 <! ������Сһ��
	cnn.AddReshape(4 * 4 * 30, 1);//��������ά�Ȳ㽫(4, 4, 30)->(4*4*30, 1, 1)
	cnn.AddFullConnect(CreateMat(4 * 4 * 30, 256), CreateMat(256, 1));//����ȫ���Ӳ�4*4*30 -> 256
	cnn.AddActivation(ReLU);//���Ӽ����ReLU
	cnn.AddFullConnect(CreateMat(256, 23), CreateMat(23, 1));//����ȫ���Ӳ�256 ->23
	cnn.AddActivation(Softmax);//���Ӽ����Softmax
	cnn.InitMethod(CreateOptimizer(Adam, 1e-2, CrossEntropy));//ע���Ż���Adam, ѧϰ��Ϊ0.02, ��ʧ����Ϊ������
}
void NetTrain(vector<Mat> &input, vector<Mat> &label, Net &net, int n, int m, int times)
{
	if (m == input.size()) {
		double acc = 0;
		for (int i = 0; i <= n; ++i) {
			acc = net.TrainModel(input, label);
			cout << "����������" << i << "\t\t��׼ȷ�ʣ�" << acc * 100 << "%" << endl;
			if (acc == 1)break;
		}
	}
	else {
		//���ɳ���������
		vector<int> randomVec;
		for (int i = 0; i < (int)input.size(); ++i)
			randomVec.push_back(i);
		//��ȷ��
		double acc = 0;
		//���
		double error = 0;
		double d_error = 0;
		//��ȷ��
		double mean_acc = 0;
		//���
		double mean_error = 0;
		//�ٷְ���ȷ�ʳ��ֵĴ���
		int count = 0;
		bool calall = false;
		for (int i = 0; i <= n; ++i) {
			//����
			random_shuffle(randomVec.begin(), randomVec.end());
			//����ѵ��������
			vector<Mat> in(m);
			//����ѵ������ǩ
			vector<Mat> out(m);
			for (int j = 0; j < m; ++j) {
				in[j] = input[randomVec[j]];
				out[j] = label[randomVec[j]];
			}
			//ѵ��ģ��
			acc = net.TrainModel(in, out, &error);
			if (i%times == 0) {				
				printf("����������%d \t������ȷ�ʣ�%0.2lf%% \t��%0.4e\n", i, acc * 100, error);
			}
		}
	}
}
void train(vector<Mat> & input, vector<Mat> & label, Net &net, int n, int m, int times, string save_path)
{
	NetTrain(input, label, net, n, m, times);
	printf("����ȷ�ʣ�%0.4lf%%\n", net.Accuracy(input, label) * 100);
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
	//ȫ��������
	CreateNN(net);
	train(input, output, net, 1000, 50, 100, save_nn_path);
	//�������
	/*CreateCNN(net);
	train(input, output, net, 300, 50, 1, save_cnn_path);*/
	//ȫ���Ӳ�������
	testNet("./train_data", save_nn_path, "nn_finish.csv");
	//�����������
	//testNet("./train_data", save_cnn_path, "cnn_finish.csv");
	end = clock();
	printf("\ntime=%fs\n", (double)(end - start) / CLK_TCK);
	system("pause");
	return 0;
}