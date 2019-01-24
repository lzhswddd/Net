#include <iostream>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <iomanip>
#include "include.h"

using namespace nn;
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;

void read_data(vector<Mat> &input, vector<Mat> &output)
{
	ifstream file("data.txt");
	if (!file.is_open())return;
	int row, col, count;
	file >> row;
	file >> col;
	file >> count;
	vector<Mat>(row).swap(input);
	vector<Mat>(row).swap(output);
	for (int i = 0; i < row; i++) {
		int index;
		output[i] = zeros(count, 1);
		input[i] = zeros(col, 1);
		file >> index;
		output[i](index) = 1;
		for (int j = 0; j < col; j++) {
			file >> index;
			input[i](j) = index;
		}
	}
	file.close();
}

void NNtrain(vector<Mat> &input, vector<Mat> &output, int n, int m, string save_path)
{		
	if (input.empty(), output.empty())return;
	Net nn;
	nn.AddFullConnect(CreateMat(25, 16*16), CreateMat(25, 1));
	nn.AddActivation(ReLU);
	nn.AddFullConnect(CreateMat(25, 25), CreateMat(25, 1));
	nn.AddActivation(ReLU);
	nn.AddFullConnect(CreateMat(23, 25), CreateMat(23, 1));
	nn.AddActivation(Softmax);
	//OptimizerP op = nn::GradientDescentOptimizer(1e-2).minimize(Quadratic);
	//OptimizerP op = nn::MomentumOptimizer(1e-2).minimize(CrossEntropy);
	//OptimizerP op = nn::NesterovMomentumOptimizer(1e-1).minimize(Quadratic);
	//OptimizerP op = nn::AdagradOptimizer(8e-1).minimize(Quadratic);
	//OptimizerP op = nn::RMSPropOptimizer(1e-2).minimize(Quadratic);
	OptimizerP op = nn::AdamOptimizer(1e-2).minimize(CrossEntropy);
	//OptimizerP op = nn::NesterovAdamOptimizer(1e-2).minimize(Quadratic);

	nn.InitMethod(op);
	if (m == input.size()) {
		double acc = 0;
		for (int i = 0; i <= n; ++i) {
			acc = nn.TrainModel(input, output);
			cout << "迭代次数：" << i << "\t\t总准确率：" << acc * 100 << "%" << endl;
			if (acc == 1)break;
		}
	}
	else {
		int times = 10;
		vector<int> randomVec;
		for (int i = 0; i < (int)input.size(); ++i)
			randomVec.push_back(i);
		double acc = 0;
		for (int i = 0; i <= n; ++i) {
			random_shuffle(randomVec.begin(), randomVec.end());
			vector<Mat> in(m);
			vector<Mat> out(m);
			for (int j = 0; j < m; ++j) {
				in[j] = input[randomVec[j]];
				out[j] = output[randomVec[j]];
			}
			acc = nn.TrainModel(in, out);
			cout << "迭代次数：" << i << "\t\t当前样本准确率：" << acc * 100 << "%";
			if (acc == 1) {
				acc = nn.Accuracy(input, output);
				cout << "\t总准确率：" << acc * 100 << "%";
				if (acc == 1) break;
			} 
			cout << endl;
		}
		acc = nn.Accuracy(input, output);
		cout << "\t总准确率：" << acc * 100 << "%";
	}
	nn.SaveModel(save_path);
}
void trainNN(int n, int m, string save_path)
{
	vector<Mat> input;
	vector<Mat> output;
	read_data(input, output);	
	if (m > 0) NNtrain(input, output, n, m, save_path);
	else NNtrain(input, output, n, (int)input.size(), save_path);
}
Mat NN_Input(Image &input)
{
	Mat mat = zeros(input.length(), 1);
	input = 255 - input;
	for (int i = 0; i < input.length(); ++i) {
		mat(i) = (double)input(i)[0];
	}
	mat /= Max(mat);
	return mat;
}
void testNet_Input(string file_path, string model_path)
{
	vector<string> files;
	getFiles("train_data", files);
	if (files.empty()) {
		printf("file is empty!\n");
		return;
	}
	string word = "ABCDEFGHJKLMNPRSTUVWXYZ";
	Net nn(model_path);
	ofstream out("finish.csv");
	if (out.is_open()) {
		for (size_t i = 0; i < files.size(); ++i) {
			Image image = Imread(files[i].c_str(), true);
			Mat output = nn.Run(NN_Input(image));
			int adr = output.maxAt();
			out << word[adr] << ',' << adr << ',' << files[i] << endl;
			cout << word[adr] << ' ' << adr << '\t';
			cout << files[i] << endl;
		}
		out.close();
	}
}
void trainCNN(int n, int m, string save_path)
{
	vector<Mat> input;
	vector<Mat> output;
	read_data(input, output);
	if (input.empty(), output.empty())return;
	Net cnn;
	cnn.AddReshape(16, 16);
	cnn.AddConv(CreateMat(5, 5, 1, 32), CreateMat(1, 1, 32));
	cnn.AddActivation(ReLU);
	cnn.AddMaxPool(2, 2);
	cnn.AddConv(CreateMat(5, 5, 32, 64), CreateMat(1, 1, 64));
	cnn.AddActivation(ReLU);
	cnn.AddMaxPool(2, 2);
	cnn.AddReshape(4 * 4 * 64, 1);
	cnn.AddFullConnect(CreateMat(4 * 4 * 64, 1024), CreateMat(1024, 1));
	cnn.AddActivation(ReLU);
	cnn.AddFullConnect(CreateMat(1024, 23), CreateMat(23, 1));
	cnn.AddActivation(Softmax);
	//OptimizerP op = GradientDescentOptimizer(1e-2).minimize(Quadratic);
	//OptimizerP op = MomentumOptimizer(1e-2).minimize(CrossEntropy);
	//OptimizerP op = NesterovMomentumOptimizer(1e-1).minimize(Quadratic);
	//OptimizerP op = AdagradOptimizer(8e-1).minimize(Quadratic);
	//OptimizerP op = RMSPropOptimizer(1e-2).minimize(Quadratic);
	OptimizerP op = AdamOptimizer(1e-2).minimize(CrossEntropy);
	//OptimizerP op = NesterovAdamOptimizer(1e-2).minimize(Quadratic);

	cnn.InitMethod(op);
	if (m == input.size()) {
		double acc = 0;
		for (int i = 0; i <= n; ++i) {
			acc = cnn.TrainModel(input, output);
			cout << "迭代次数：" << i << "\t\t总准确率：" << acc * 100 << "%" << endl;
			if (acc == 1)break;
		}
	}
	else {
		int times = 10;
		vector<int> randomVec;
		for (int i = 0; i < (int)input.size(); ++i)
			randomVec.push_back(i);
		double acc = 0;
		for (int i = 0; i <= n; ++i) {
			random_shuffle(randomVec.begin(), randomVec.end());
			vector<Mat> in(m);
			vector<Mat> out(m);
			for (int j = 0; j < m; ++j) {
				in[j] = input[randomVec[j]];
				out[j] = output[randomVec[j]];
			}
			acc = cnn.TrainModel(in, out);
			cout << "迭代次数：" << i << "\t\t当前样本准确率：" << acc * 100 << "%";
			if (acc == 1) {
				acc = cnn.Accuracy(input, output);
				cout << "\t总准确率：" << acc * 100 << "%";
				if (acc == 1) break;
			}
			cout << endl;
		}
		acc = cnn.Accuracy(input, output);
		cout << "\t总准确率：" << acc * 100 << "%";
	}
	cnn.SaveModel(save_path);
}
int main()
{
	clock_t start, end;
	start = clock();
	Srandom();
	string save_nn_path = "./model/nn_model.conf";
	string save_cnn_path = "./model/cnn_model.conf";
	//trainNN(5000, 100, save_nn_path);
	trainCNN(5000, 100, save_cnn_path);
	//testNet_Input("./train_data", save_nn_path);
	testNet_Input("./train_data", save_cnn_path);
	end = clock();
	printf("\ntime=%fs\n", (double)(end - start) / CLK_TCK);
	system("pause");
	return 0;
}