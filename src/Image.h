#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "Vriable.h"
#include "Mat.h"

namespace nn {	
	class Image
	{
	public:
		explicit Image();
		Image(const char *image_path);
		Image(int rows, int cols, int channels = 1);
		Image(uchar *data, int rows, int cols, int channels);
		Image(const Image &src);
		~Image();
		int length()const;
		bool empty()const;
		void swap(Image &src)const; 
		void operator = (const Image &src);
		const Image operator + (const uchar value)const;
		const Image operator + (const Image &image)const;
		void operator += (const Image &image);
		void operator += (const uchar value);
		friend const Image operator + (const uchar value, const Image &src);
		const Image operator - (const uchar value)const; 		
		const Image operator - (const Image &image)const;
		void operator -= (const Image &image);
		void operator -= (const uchar value);
		friend const Image operator - (const uchar value, const Image &src);
		Vec<uchar> operator () (const int row, const int col)const; 
		Vec<uchar> operator () (const int index)const;
		uchar& operator () (const int row, const int col, const int channel)const;	
		int cols;
		int rows;
		int channels;
		uchar *data;

	protected:
		void copy(const Image &src);
	};

	const Image toImage(const Mat &src);
	const Image Imread(const char *image_path, bool is_gray = false);
	void Imwrite(const char *image_path, Image& image);
	void RGB2Gray(const Image& src, Image& dst);
	//void Gray2RGB(const Image& src, Image& dst);
}

#endif //__IMAGE_H__