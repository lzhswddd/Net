#include <memory.h>
#include <iostream>
#include "Image.h"
#include "Vriable.h"
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define TJE_IMPLEMENTATION
#include "tiny_jpeg.h"

using namespace nn;
using namespace std;

Image::Image() : data(nullptr), rows(0), cols(0), channels(0) {}

Image::Image(const char * image_path) : data(nullptr), rows(0), cols(0), channels(0)
{
	*this = Imread(image_path);
}

nn::Image::Image(int rows, int cols, int channels) : data(new uchar[rows*cols*channels]), rows(rows), cols(cols), channels(channels)
{
	memset(data,0,sizeof(uchar)*rows*cols*channels);
}

Image::Image(uchar * data, int rows, int cols, int channels) : data(new uchar[rows*cols*channels]), rows(rows), cols(cols), channels(channels) 
{
	memcpy(this->data, data, sizeof(uchar)*rows*cols*channels);
}

Image::Image(const Image & src): data(nullptr), rows(0), cols(0), channels(0)
{
	copy(src);
}


Image::~Image()
{
	if (data != nullptr) {
		delete[]data;
		data = nullptr;
	}
	cols = 0;
	rows = 0;
	channels = 0;
}

int Image::length() const
{
	return rows*cols*channels;
}

bool Image::empty() const
{
	return (data == nullptr);
}

void Image::swap(Image & src)const
{
	src = *this;
}

const Image Image::operator+(const uchar value) const
{
	if (data == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw Image();
	}
	Image mark(*this);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			for (int k = 0; k < channels; k++)
				mark(i, j, k) = mark(i, j, k) + value;
	return mark;
}

const Image Image::operator-(const uchar value) const
{
	if (data == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw Image();
	}
	Image mark(*this);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			for (int k = 0; k < channels; k++)
				mark(i, j, k) = mark(i, j, k) - value;
	return mark;
}

const Image Image::operator+(const Image & image) const
{
	if (data == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw Image();
	}
	if (rows != image.rows || cols != image.cols || channels != image.channels) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw Image();
	}
	Image mark(*this);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			for (int k = 0; k < channels; k++)
				mark(i, j, k) = mark(i, j, k) + image(i, j, k);
	return mark;
}

const Image Image::operator-(const Image & image) const
{
	if (data == nullptr) {
		cerr << errinfo[ERR_INFO_EMPTY] << endl;
		throw Image();
	}
	if (rows != image.rows || cols != image.cols || channels != image.channels) {
		cerr << errinfo[ERR_INFO_SIZE] << endl;
		throw Image();
	}
	Image mark(*this);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			for (int k = 0; k < channels; k++)
				mark(i, j, k) = mark(i, j, k) - image(i, j, k);
	return mark;
}

void Image::operator+=(const Image & image)
{
	*this = *this + image;
}

void Image::operator-=(const Image & image)
{
	*this = *this - image;
}

void Image::operator+=(const uchar value)
{
	*this = *this + value;
}

void Image::operator-=(const uchar value)
{
	*this = *this - value;
}

Vec<uchar> Image::operator()(const int row, const int col) const
{
	if (row < 0 || row >= rows || col < 0 || col >= cols) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return Vec<uchar>(data + row * cols*channels + col * channels);
}

Vec<uchar> Image::operator()(const int index) const
{
	if (index < 0 || index >= rows * cols*channels) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return Vec<uchar>(data + index * channels);

}

uchar & Image::operator()(const int row, const int col, const int channel) const
{
	if (row < 0 || row >= rows || col < 0 || col >= cols || channel < 0 || channel >= channels) {
		fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
	}
	return data[row*cols*channels + col * channels + channel];
}

void Image::operator = (const Image & src)
{
	copy(src);
}

void Image::copy(const Image & src)
{
	if (src.data != nullptr) {
		if (data != nullptr) {
			delete[] data;
			data = nullptr;
		}
		rows = src.rows;
		cols = src.cols;
		channels = src.channels;
		data = new uchar[rows*cols*channels];
		memcpy(data, src.data, sizeof(uchar)*rows*cols*channels);
	}
}

const Image nn::operator+(const uchar value, const Image &src)
{
	return src + value;
}

const Image nn::operator-(const uchar value, const Image & src)
{
	return src - value;
}

const Mat nn::Image2Mat(const Image & src)
{
	if (src.empty())return Mat();
	Mat mat(src.rows, src.cols, src.channels);
	for (int i = 0; i < mat.rows(); ++i)
		for (int j = 0; j < mat.cols(); ++j)
			for (int z = 0; z < mat.channels(); ++z)
				mat(i, j, 0) = (uchar)src(i, j, z);
	return mat;
}

const Image nn::Mat2Image(const Mat & src)
{
	if(src.empty())return Image();
	Image image;
	image.rows = src.rows();
	image.cols = src.cols();
	image.channels = src.channels();;
	image.data = new uchar[image.rows*image.cols*image.channels];
	for (int i = 0; i < src.rows(); ++i)
		for (int j = 0; j < src.cols(); ++j)
			for (int z = 0; z < src.channels(); ++z)
				image(i, j, z) = (uchar)src(i, j, z);
	return image;
}

const Image nn::Imread(const char * image_path, bool is_gray)
{
	Image image;
	image.data = stbi_load(image_path, &image.cols, &image.rows, &image.channels, 0);
	if (image.data == nullptr) {
		fprintf(stderr, "load %s fail.\n", image_path);
		throw "load image fail";
	}
	if (is_gray) {
		RGB2Gray(image, image);
	}
	return image;
}

const Mat nn::mImread(const char * image_path, bool is_gray)
{
	return Image2Mat(Imread(image_path, is_gray));
}

void nn::Imwrite(const char * image_path, Image & image)
{
	if (!tje_encode_to_file(image_path, image.cols, image.rows, image.channels, true, image.data)) {
		fprintf(stderr, "save %s fail.\n", image_path);
		throw "save image fail";
	}
}

void nn::Imwrite(const char * image_path, const Mat & image)
{
	Image img = Mat2Image(image);
	if (!tje_encode_to_file(image_path, img.cols, img.rows, img.channels, true, img.data)) {
		fprintf(stderr, "save %s fail.\n", image_path);
		throw "save image fail";
	}
}

void nn::RGB2Gray(const Image & src, Image & dst)
{
	if (src.empty())return;
	if (src.channels != 1) {
		Image img = Mat2Image(zeros(src.rows, src.cols));
		for (int i = 0; i < src.rows; ++i) {
			for (int j = 0; j < src.cols; ++j) {
				Vec<uchar> rgb = src(i, j);
				img(i, j, 0) = (uchar)(((int)rgb[0] * 30 + (int)rgb[1] * 59 + (int)rgb[2] * 11 + 50) / 100);
			}
		}
		img.swap(dst);
	}
	else {
		src.swap(dst);
	}
}

