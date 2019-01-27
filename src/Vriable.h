#ifndef __VARIABLE_H__
#define __VARIABLE_H__
#include <stdio.h>
namespace nn
{
	static const double pi = 3.1415926535897932384626433832795f;
	typedef unsigned char uchar;
	typedef unsigned int uint;
	/**
	CONV2D			�����
	MAX_POOL		���ֵ�ػ���
	AVERAGE_POOL	ƽ��ֵ�ػ���
	FULL_CONNECTION ȫ���Ӳ�
	ACTIVATION		�����
	RESHAPE			����ά�Ȳ�
	DROPOUT			���ʹ�ܲ�
	*/
	enum LayerType {
		CONV2D = 0,
		MAX_POOL,
		AVERAGE_POOL,
		FULL_CONNECTION,
		ACTIVATION,
		RESHAPE,
		DROPOUT
	};
	/**
	None				���ṩ�Ż�����
	GradientDescent		�ṩ�ݶ��½���
	Momentum			�ṩ�����ݶ��½���
	NesterovMomentum	�ṩԤ�⶯���ݶ��½���
	Adagrad				�ṩ����Ӧѧϰ���ݶ��½���
	RMSProp				�ṩ��������Ӧѧϰ���ݶ��½���
	Adam				�ṩ����Ӧѧϰ�ʶ����ݶ��½���
	NesterovAdam		�ṩ����Ӧѧϰ��Ԥ�⶯���ݶ��½���
	*/
	enum OptimizerMethod
	{
		None= 0,		//!< ���ṩ�Ż�����
		GradientDescent,//!< �ṩ�ݶ��½���
		Momentum,		//!< �ṩ�����ݶ��½���
		NesterovMomentum,//!< �ṩԤ�⶯���ݶ��½���
		Adagrad,		//!< �ṩ����Ӧѧϰ���ݶ��½���
		RMSProp,		//!< �ṩ��������Ӧѧϰ���ݶ��½���
		Adam,			//!< �ṩ����Ӧѧϰ�ʶ����ݶ��½���
		NesterovAdam	//!< �ṩ����Ӧѧϰ��Ԥ�⶯���ݶ��½���
	};
	enum BorderTypes {
		BORDER_CONSTANT = 0, //!< `iiiiii|abcdefgh|iiiiiii`  with some specified `i`
		BORDER_REPLICATE = 1, //!< `aaaaaa|abcdefgh|hhhhhhh`
		BORDER_REFLECT = 2, //!< `fedcba|abcdefgh|hgfedcb`
		BORDER_WRAP = 3, //!< `cdefgh|abcdefgh|abcdefg`
		BORDER_REFLECT_101 = 4, //!< `gfedcb|abcdefgh|gfedcba`
		BORDER_TRANSPARENT = 5, //!< `uvwxyz|abcdefgh|ijklmno`
		BORDER_ISOLATED = 16 //!< do not look outside of ROI
	};
	enum MatErrorInfo {
		ERR_INFO_EMPTY = 0,
		ERR_INFO_SQUARE,
		ERR_INFO_ADJ,
		ERR_INFO_INV,
		ERR_INFO_POW,
		ERR_INFO_IND,
		ERR_INFO_CON,
		ERR_INFO_EIGEN,
		ERR_INFO_LEN,
		ERR_INFO_MEMOUT,
		ERR_INFO_UNLESS,
		ERR_INFO_SIZE,
		ERR_INFO_MULT,
		ERR_INFO_NORM,
		ERR_INFO_VALUE,
		ERR_INFO_PINV,
		ERR_INFO_DET,
		ERR_INFO_DIM,
	};
	static const char *errinfo[] = {
		"error 0: ����Ϊ��!",
		"error 1: �����Ƿ���",
		"error 2: �����Ƿ��󣬲������ð������",
		"error 3: �����Ƿ��󣬲������������",
		"error 4: �����Ƿ��󣬲��ܽ��д������㣡",
		"error 5: �����Ƿ��󣬲�������Ϊ��λ����",
		"error 6: ��������!",
		"error 7: ����û��ʵ������ֵ!",
		"error 8: ����ά��Ϊ0��",
		"error 9: �����������磡",
		"error 10: ����������Ч��",
		"error 11: ��������ά�Ȳ�һ�£�",
		"error 12: ��������ά�Ȳ�����˷�������",
		"error 13: ����ά�Ȳ�Ϊ1������������",
		"error 14: ����Υ����",
		"error 15: ���������ʧ�ܣ�"
		"error 16: ����ʽΪ0��",
		"error 17: ��֧����ά������"
	};
	enum X_Y_Z {
		ROW = 0,
		COL,
		CHANNEL
	};
	enum direction {
		LEFT = 0,
		RIGHT
	};
	class Size
	{
	public:
		Size() :hei(0), wid(0) {}
		Size(int height, int width) :hei(height), wid(width) {}
		~Size() {}
		int hei;
		int wid;
	};
	class Size3
	{
	public:
		explicit Size3() : x(0), y(0), z(0) {}
		Size3(int x, int y, int z = 1) : x(x), y(y), z(z) {}
		int x;
		int y;
		int z;
	};
	template<class Tp_>
	class Point2
	{
	public:
		Point2() :x(), y() {}
		Point2(Tp_ x, Tp_ y) :x(x), y(y) {}
		~Point2() {}
		bool operator == (const Point2<Tp_> &P)const
		{
			return (x == P.x) && (y == P.y);
		}
		bool operator != (const Point2<Tp_> &P)const
		{
			return !((*this) == P);
		}
		Tp_ x;
		Tp_ y;		
	};
	template<typename Tp_, typename T2>
	const Point2<Tp_> operator + (const Point2<Tp_> &P, const T2& v)
	{
		return Point2<Tp_>(P.x + v, P.y + v);
	}
	template<typename Tp_>
	const Point2<Tp_> operator + (const Point2<Tp_> &P1, const Point2<Tp_>& P2)
	{
		return Point2<Tp_>(P1.x + P2.x, P1.y + P2.y);
	}
	template<typename Tp_, typename T2>
	const Point2<Tp_> operator - (const Point2<Tp_> &P, const T2& v)
	{
		return Point2<Tp_>(P.x - v, P.y - v);
	}
	template<typename Tp_>
	const Point2<Tp_> operator - (const Point2<Tp_> &P1, const Point2<Tp_>& P2)
	{
		return Point2<Tp_>(P1.x - P2.x, P1.y - P2.y);
	}
	template<typename Tp_, typename T2>
	const Point2<Tp_> operator * (const Point2<Tp_> &P, const T2& v)
	{
		return Point2<Tp_>(P.x * v, P.y * v);
	}
	template<typename Tp_>
	const Tp_ operator * (const Point2<Tp_> &P1, const Point2<Tp_>& P2)
	{
		return P1.x * P2.x + P1.y * P2.y;
	}
	typedef Point2<char> Point2c;
	typedef Point2<uchar> Point2Uc;
	typedef Point2<int> Point2i;
	typedef Point2<uint> Point2Ui;
	typedef Point2<float> Point2f;
	typedef Point2<double> Point2d;
	typedef Point2i Point;

	template<class Tp_>
	class Point3
	{
	public:
		Point3() :x(), y(), z() {}
		Point3(Tp_ x, Tp_ y, Tp_ z) :x(x), y(y), z(z) {}
		~Point3() {}
		bool operator == (Point2<Tp_> &P)const
		{
			return (x == P.x) && (y == P.y) && (z == P.z);
		}
		bool operator != (Point2<Tp_> &P)const
		{
			return !((*this) == P);
		}
		Tp_ x;
		Tp_ y;
		Tp_ z;
	};
	template<typename Tp_, typename T2>
	const Point3<Tp_> operator + (const Point3<Tp_> &P, const T2& v)
	{
		return Point3<Tp_>(P.x + v, P.y + v, P.z + v);
	}
	template<typename Tp_>
	const Point3<Tp_> operator + (const Point3<Tp_> &P1, const Point3<Tp_>& P2)
	{
		return Point3<Tp_>(P1.x + P2.x, P1.y + P2.y, P1.z + P2.z);
	}
	template<typename Tp_, typename T2>
	const Point3<Tp_> operator - (const Point3<Tp_> &P, const T2& v)
	{
		return Point3<Tp_>(P.x - v, P.y - v, P.z + v);
	}
	template<typename Tp_>
	const Point3<Tp_> operator - (const Point3<Tp_> &P1, const Point3<Tp_>& P2)
	{
		return Point2<Tp_>(P1.x - P2.x, P1.y - P2.y, P1.z - P2.z);
	}
	template<typename Tp_, typename T2>
	const Point3<Tp_> operator * (const Point3<Tp_> &P, const T2& v)
	{
		return Point3<Tp_>(P.x * v, P.y * v, P.z * v);
	}
	template<typename Tp_>
	const Tp_ operator * (const Point3<Tp_> &P1, const Point3<Tp_>& P2)
	{
		return P1.x * P2.x + P1.y * P2.y + P1.z * P2.z;
	}
	typedef Point3<char> Point3c;
	typedef Point3<uchar> Point3Uc;
	typedef Point3<int> Point3i;
	typedef Point3<uint> Point3Ui;
	typedef Point3<float> Point3f;
	typedef Point3<double> Point3d;

	template<class Type>
	class Vec
	{
	public:
		explicit Vec() {
			row = 0;
			col = 0;
			channel = 0;
			data = nullptr;
		}
		Vec(Type *data, int size = 3) : row(size), col(1), channel(1) {
			this->data = data;
		}
		void release()
		{
			delete[] data;
			data = nullptr;
			row = 0;
			col = 0;
			channel = 0;
		}
		Type& operator [](const int index)const {
			if (index < 0 || index >= row*col*channel) {
				fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
			}
			return data[index];
		}
		Type& operator ()(const int x, const int y)const {
			if (x < 0 || x >= row || y < 0 || y >= col) {
				fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
			}
			return data[(x*col + y)*channel];
		}
		Type& operator ()(const int x, const int y, const int z)const {
			if (x < 0 || x >= row || y < 0 || y >= col|| z < 0 || z >= channel) {
				fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
			}
			return data[(x*col + y)*channel + z];
		}
	private:
		int row;
		int col;
		int channel;
		Type* data;
	};
}

#endif //__VARIABLE_H__
