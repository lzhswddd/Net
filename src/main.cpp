#include <iostream>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <iomanip>
#include <io.h>
#include <direct.h>
#include <Windows.h>
#include "include.h"

using namespace nn;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

LARGE_INTEGER t1, t2, start, end, tc;

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
	nn.AddFullConnect(CreateMat(16 * 16, 50), CreateMat(50, 1));//����ȫ���� 16*16 -> 50
	nn.AddActivation(Sigmoid);//���Ӽ����Sigmoid
	//nn.AddDropout(0.3);//����dropout��, ������Ϊ0.3
	nn.AddFullConnect(CreateMat(50, 50), CreateMat(50, 1));//����ȫ���� 50 -> 50
	nn.AddActivation(Sigmoid);//���Ӽ����Sigmoid
	nn.AddFullConnect(CreateMat(50, 23), CreateMat(23, 1));//����ȫ���� 50 -> 23
	nn.AddActivation(Softmax);//���Ӽ����Softmax
	nn.InitMethod(CreateOptimizer(Momentum, 7e-1, Quadratic));//ע���Ż���Momentum, ѧϰ��Ϊ0.5, ��ʧ����Ϊƽ����
}
void CreateCNN(Net & cnn)
{
	cnn.AddReshape(16, 16, 1);//��������ά�Ȳ㽫(16*16, 1, 1)->(16, 16, 1)
	cnn.AddConv(CreateMat(5, 5, 1, 10), CreateMat(1, 1, 10));//���Ӿ����(�����Ϊ3*3*3
	cnn.AddActivation(ELU);//���Ӽ����ELU
	cnn.AddMaxPool(2, 2);//�������ֵ�ػ���(�ػ���СΪ2,2 <! ������Сһ��
	cnn.AddConv(CreateMat(3, 3, 10, 10), CreateMat(1, 1, 10));//���Ӿ����(�����Ϊ3*3*3
	cnn.AddActivation(ELU);//���Ӽ����ELU
	cnn.AddMaxPool(2, 2);//�������ֵ�ػ���(�ػ���СΪ2,2 <! ������Сһ��
	cnn.AddConv(CreateMat(3, 3, 10, 20), CreateMat(1, 1, 20));//���Ӿ����(�����Ϊ3*3*3
	cnn.AddActivation(ELU);//���Ӽ����ELU
	cnn.AddConv(4, 4, 20, 23, false, Size(1, 1), Point(0, 0));//���Ӿ����(�����Ϊ3*3*3
	cnn.AddReshape(1 * 1 * 23, 1, 1);//��������ά�Ȳ㽫(4, 4, 9)->(4*4*9, 1, 1)
	//cnn.AddReshape(4 * 4 * 10, 1, 1);//��������ά�Ȳ㽫(4, 4, 9)->(4*4*9, 1, 1)
	//cnn.AddFullConnect(CreateMat(4 * 4 * 10, 128), CreateMat(128, 1));//����ȫ���Ӳ�4*4*9 -> 128
	//cnn.AddActivation(ELU);//���Ӽ����ELU
	//cnn.AddFullConnect(CreateMat(128, 23), CreateMat(23, 1));//����ȫ���Ӳ�128 ->23
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
		//ƽ�����
		double mean_error = (double)times;
		QueryPerformanceCounter(&start);
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

			acc += net.TrainModel(in, out, &error);
			mean_error += error;
			if (i%times == 0) {		
				QueryPerformanceCounter(&end);
				printf("����������%d \t%d��ƽ��������ȷ�ʣ�%0.2lf%% \tƽ����%0.4e\t��ʱ��%0.4lfs\tԤ����ʱ��%0.4lfs\n", i, times, acc * 100 / (double)times, mean_error / (double)times, (end.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart, (end.QuadPart - start.QuadPart)*1.0 / tc.QuadPart / double(times)* n);
				QueryPerformanceCounter(&start);
				mean_error = acc = 0;
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


//#define NET_CNN
#ifndef NET_CNN
#define NET_NN
#endif // NET_CNN


int main(int argc, char *argv[])
{
	{
		QueryPerformanceFrequency(&tc);		
		if (0 != _access("data.txt", 0))
		{
			system("python totxt.py");
		}
		vector<Mat> input;
		vector<Mat> output;
		read_data(input, output);
		Srandom();
		string save_nn_path = "./model/nn_model.conf";
		string save_nn_output = "nn_finish.csv";
		string save_cnn_path = "./model/cnn_model.conf";
		string save_cnn_output = "cnn_finish.csv";
		string save_path, save_output;
		Net net; int batchs_num, iteration, show_times;
#ifdef NET_NN
		//ȫ��������
		CreateNN(net);
		iteration = 3000; batchs_num = 50; show_times = 10;
		save_path = save_nn_path; save_output = save_nn_output;
#else
		//�������
		CreateCNN(net);
		iteration = 500; batchs_num = 50; show_times = 5;
		save_path = save_cnn_path; save_output = save_cnn_output;

#endif //NET_NN
		QueryPerformanceCounter(&t1);
		//ѵ������
		train(input, output, net, iteration, batchs_num, show_times, save_path);
		QueryPerformanceCounter(&t2);
		printf("Use Time:%f\n", (t2.QuadPart - t1.QuadPart)*1.0 / tc.QuadPart);
		printf("����������ʼ����!\n");
		getchar();
		//��������
		testNet("./train_data", save_path, save_output);
	}
	system("pause");
	return 0;
}