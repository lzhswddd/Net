#ifndef __MATRIX_H__
#define __MATRIX_H__
#include <vector>
#include "Vriable.h"
#define DLength(x) (sizeof(x)/sizeof(double))
#define ILength(x) (sizeof(x)/sizeof(int))

namespace nn {
	class MatCommaInitializer_;
	class Matrix
	{
	public:
		explicit Matrix();
		/**
		@brief ����n*n*1�ķ���,��0���
		@param n ��������
		*/
		Matrix(int n);
		/**
		@brief ����row*col*1�ķ���,��0���
		@param row ��������
		@param col ��������
		*/
		Matrix(int row, int col);
		/**
		@brief ����row*col*depth�ķ���,��0���
		@param row ��������
		@param col ��������
		@param depth ����ͨ����
		*/
		Matrix(int row, int col, int depth);
		/**
		@brief ����size*1�ķ���,��0���
		@param size_ ����ߴ�
		*/
		Matrix(Size size_);
		/**
		@brief ����size�ķ���,��0���
		@param size_ ����ߴ�
		*/
		Matrix(Size3 size_);
		/**
		@brief ��������
		@param src ��������
		*/
		Matrix(const Matrix *src);
		/**
		��������
		@param src ��������
		*/
		Matrix(const Matrix &src);
		/**
		@brief ������a��b�ϲ�(COLΪ���кϲ�|ROWΪ���кϲ�)
		@param a �������1
		@param b �������2
		@param merge �ϲ���ʽ
		*/
		Matrix(Matrix a, Matrix b, X_Y_Z merge);
		/**
		@brief ���캯��
		���m
		@param m ����
		*/
		Matrix(MatCommaInitializer_ &m);
		/**
		@brief ����n*n*1�ķ���,Ԫ��Ϊmatrix
		@param matrix ����Ԫ��
		@param n �����С
		*/
		Matrix(int *matrix, int n);
		/**
		@brief ����n*n*1�ķ���,Ԫ��Ϊmatrix
		@param matrix ����Ԫ��
		@param n �����С
		*/
		Matrix(double *matrix, int n);
		/**
		@brief ����row*col*1�ķ���,Ԫ��Ϊmatrix
		@param matrix ����Ԫ��
		@param row ��������
		@param col ��������
		*/
		Matrix(int *matrix, int row, int col, int channel = 1);
		/**
		@brief ����row*col*1�ķ���,Ԫ��Ϊmatrix
		@param matrix ����Ԫ��
		@param row ��������
		@param col ��������
		*/
		Matrix(double *matrix, int row, int col, int channel = 1);
		~Matrix();
		/**
		@brief ���ؾ���ָ��
		*/
		double* mat_()const;
		/**
		@brief ���ά��
		*/
		void DimCheck()const;
		/**
		@brief ���ؾ���ߴ�(row,col,channel)
		*/
		Size3 size3()const;
		/**
		@brief ���ؾ�������
		*/
		int rows()const;
		/**
		@brief ���ؾ�������
		*/
		int cols()const;
		/**
		@brief ���ؾ���ͨ����
		*/
		int channels()const;
		/**
		@brief ���ؾ����С(row*col*1)
		*/
		int size()const;
		/**
		@brief ���ؾ����СSize(row,col)
		*/
		Size mSize()const;
		/**
		@brief ���ؾ����С(row*col*1)
		*/
		int length()const;
		/**
		@brief ���ؾ���״̬
		0Ϊ����
		-1Ϊ�վ���
		-2Ϊ�Ƿ���
		*/
		int isEnable()const;
		/**
		@brief ���ؾ����Ƿ�Ϊ��
		*/
		bool empty()const;
		/**
		@brief ���ؾ����Ƿ�Ϊ����
		*/
		bool Square()const;
		/**
		@brief ���������ؾ���Ԫ��
		@param index ����
		*/
		double& at(int index)const;
		/**
		@brief ���������ؾ���Ԫ��
		@param index_x ������
		@param index_y ������
		*/
		double& at(int index_y, int index_x)const;
		/**
		@brief ������ת��Ϊ��Ӧ����������
		@param index ����
		*/
		int toX(int index)const;
		/**
		@brief ������ת��Ϊ��Ӧ����������
		@param index ����
		*/
		int toY(int index)const;

		/**
		@brief �����һ��Ԫ��
		*/
		double frist()const;
		/**
		@brief ���ؾ�����value��ȵĵ�һ��Ԫ������
		@param value Ԫ��
		*/
		int find(double value)const;
		/**
		@brief ���ؾ���Ԫ�����ֵ������
		*/
		int maxAt()const;
		/**
		@brief ���ؾ���Ԫ����Сֵ������
		*/
		int minAt()const;
		/**
		@brief ���ؾ����Ƿ����value
		@param value Ԫ��
		*/
		bool contains(double value)const;
		/**
		@brief ���ؾ�����value��ȵĵ�һ��Ԫ��
		@param value Ԫ��
		*/
		double& findAt(double value)const;
		/**
		@brief ���ؾ���Ԫ�����ֵ
		*/
		double& findmax()const;
		/**
		@brief ���ؾ���Ԫ����Сֵ
		*/
		double& findmin()const;
		/**
		@brief �������������򿽱�Ԫ�ص�src������
		@param src ����������
		@param Row_Start ��ȡ�г�ʼ����ֵ
		@param Row_End ��ȡ�н�������ֵ
		@param Col_Start ��ȡ�г�ʼ����ֵ
		@param Col_End ��ȡ�н�������ֵ
		*/
		void copy(Matrix &src, int Row_Start, int Row_End, int Col_Start, int Col_End)const;
		/**
		@brief �����󿽱���src
		@param src ����������
		*/
		void swap(Matrix &src)const;
		/**
		@brief �ھ�������߻����ұ����һ��1
		@param dire ѡ����ӷ�ʽ
		*/
		void addones(direction dire = LEFT);
		/**
		@brief mChannel ��src���ǵ���channelͨ��
		@param src ����
		@param channel ͨ����
		*/
		void mChannel(const Matrix &src, int channel);
		/**
		@brief ���þ���ά��
		������ı���󳤶�
		*/
		void reshape(int row,int col, int channel);
		/**
		@brief ���þ����С
		�������ԭ��С������row*col*1��Ԫ��ȫ������Ϊ0
		@param row ��������
		@param col ��������
		*/
		bool setSize(int row, int col);
		/**
		@brief ��������src
		@param src ��������
		*/
		void setvalue(const Matrix &src);
		/**
		@brief �޸ľ����Ӧ����Ԫ��
		@param number Ԫ��
		@param index ����
		*/
		void setNum(double number, int index);
		/**
		@brief �޸ľ����Ӧ����Ԫ��
		@param number Ԫ��
		@param index_y ������
		@param index_x ������
		*/
		void setNum(double number, int index_y, int index_x);
		/**
		@brief ���þ���
		@param mat ����Ԫ��
		@param row ��������
		@param col ��������
		*/
		void setMat(double *mat, int row, int col);
		/**
		@brief ���������
		*/
		void setInv();
		/**
		@brief ���þ����num����
		@param num ����
		*/
		void setPow(int num);
		/**
		@brief ����ȡ��
		*/
		void setOpp();
		/**
		@brief ���õ�λ����
		*/
		void setIden();
		/**
		@brief ���ð������
		*/
		void setAdj();
		/**
		@brief ����ת�þ���
		*/
		void setTran();

		/**
		@brief �������������
		*/
		void show()const;

		/**
		@brief ����ȡ������
		*/
		const Matrix Opp()const;
		/**
		@brief ���ؾ���ֵ����
		*/
		const Matrix Abs()const;
		/**
		@brief ���ذ�num���ݾ���
		@param num ����
		*/
		const Matrix Pow(int num)const;
		/**
		@brief ���ذ�Ԫ��ȡָ������
		*/
		const Matrix Exp()const;
		/**
		@brief ���ذ�Ԫ��ȡ��������
		*/
		const Matrix Log()const;
		/**
		@brief ���ذ�Ԫ��ȡ��������
		*/
		const Matrix Sqrt()const;
		/**
		@brief ���ذ������
		*/
		const Matrix Adj()const;
		/**
		@brief ����ת�þ���
		*/
		const Matrix Tran()const;
		/**
		@brief ���������
		*/
		const Matrix Inv()const;
		/**
		@brief �����������
		�������������
		*/
		const Matrix Reverse()const;
		const Matrix EigenvectorsMax(double offset = 1e-8)const;

		/**
		@brief ��������ʽ
		*/
		double Det();
		/**
		@brief ����num����
		@param num ������
		*/
		double Norm(int num = 1)const;
		/**
		@brief ���ض�Ӧ����������ʽ
		@param x ������
		@param y ������
		*/
		double Cof(int x, int y);
		double EigenvalueMax(double offset = 1e-8)const;
		/**
		@brief ���������ȡ�ľ���Ԫ��
		*/
		double RandSample();
		/**
		@brief ���ؾ���Ԫ�غ�
		@param num ���ô���
		@param _abs �Ƿ�ȡ����ֵ
		*/
		double Sum(int num = 1, bool _abs = false)const;
		/**
		@brief ���������+
		��ӦԪ�����
		*/
		const Matrix operator + (const double val)const;
		/**
		@brief ���������+
		��ӦԪ�����
		*/
		const Matrix operator + (const Matrix &a)const;
		/**
		@brief ���������+=
		��Ԫ�����
		*/
		void operator += (const double val);
		/**
		@brief ���������+=
		��Ԫ�����
		*/
		void operator += (const Matrix &a);
		/**
		@brief ��Ԫ���������+
		��Ԫ�����
		*/
		friend const Matrix operator + (const double value, const Matrix &mat);
		/**
		@brief ���������-
		��Ԫ��ȡ�෴��
		*/
		const Matrix operator - (void)const;
		/**
		@brief ���������-
		��Ԫ�����
		*/
		const Matrix operator - (const double val)const;
		/**
		@brief ���������-
		��ӦԪ�����
		*/
		const Matrix operator - (const Matrix &a)const;
		/**
		@brief ���������-=
		��Ԫ�����
		*/
		void operator -= (const double val);
		/**
		@brief ���������-=
		��ӦԪ�����
		*/
		void operator -= (const Matrix &a);
		/**
		@brief ��Ԫ���������-
		��Ԫ�����
		*/
		friend const Matrix operator - (const double value, const Matrix &mat);
		/**
		@brief ���������*
		��Ԫ�����
		*/
		const Matrix operator * (const double val)const;
		/**
		@brief ���������*
		��ӦԪ�����
		*/
		const Matrix operator * (const Matrix &a)const;
		/**
		@brief ���������*=
		��Ԫ�����
		*/
		void operator *= (const double val);
		/**
		@brief ���������*=
		��ӦԪ�����
		*/
		void operator *= (const Matrix &a);
		/**
		@brief ��Ԫ���������*
		��Ԫ�����
		*/
		friend const Matrix operator * (const double value, const Matrix &mat);
		/**
		@brief ���������/
		��Ԫ�����
		*/
		const Matrix operator / (const double val)const;
		/**
		@brief ���������/
		����˷�
		*/
		const Matrix operator / (const Matrix &a)const;
		/**
		@brief ���������/=
		��Ԫ�����
		*/
		void operator /= (const double val);
		/**
		@brief ���������/=
		��ӦԪ�����
		*/
		void operator /= (const Matrix &a);
		/**
		@brief ��Ԫ���������/
		��Ԫ�����
		*/
		friend const Matrix operator / (const double value, const Matrix &mat);
		/**
		@brief ���������=
		���
		*/
		void operator = (const Matrix &temp);
		/**
		@brief ���������==
		�жϾ����Ƿ����
		*/
		bool operator == (const Matrix &a)const;
		/**
		@brief ���������!=
		�жϾ����Ƿ����
		*/
		bool operator != (const Matrix &a)const;
		/**
		@brief ���ض�Ӧ����Ԫ��
		@param index ����
		*/
		double& operator () (const int index)const;
		/**
		@brief ���ض�Ӧ����Ԫ��
		@param row ������
		@param col ������
		*/
		double& operator () (const int row, const int col)const;
		/**
		@brief ���ض�Ӧ����Ԫ��
		@param row ������
		@param col ������
		@param depth ͨ������
		*/
		double& operator () (const int row, const int col, const int depth)const;
		/**
		@brief ���ؾ����Ӧ�������л���
		@param index ����
		@param rc ������ʽ
		*/
		const Matrix operator () (const int index, X_Y_Z rc)const;
		/**
		@brief ���ؾ����Ӧͨ������
		@param channel ͨ������
		*/
		const Matrix operator [] (const int channel)const;
		friend std::ostream & operator << (std::ostream &out, const Matrix &ma);

	private:
		int row;
		int col;
		int depth;
		bool square;
		double *matrix;

		void init();
		void checkSquare();
		void checkindex(int index)const;
		void checkindex(int index_x, int index_y)const;
	};
	typedef Matrix Mat;
	/**
	@brief Mat_ ������
	�̳�Mat�࣬����ʵ��
	Mat mat = (Mat_(3, 3) << 
		-1, -1, -1,
		-1,  9, -1,
		-1, -1, -1);
	*/
	class Mat_ : public Mat
	{
	public:
		explicit Mat_() {}
		/**
		@brief ����row*col*channel�ķ���,��0���
		@param row ��������
		@param col ��������
		@param depth ����ͨ����
		*/
		Mat_(int row, int col = 1, int channel = 1) : Mat(row, col, channel) {}
		/**
		@brief ����size_[0]*size_[1]*size_[2]�ķ���,��0���
		���ͷ�Vec<int> size_���ڴ�
		@param size_ ����ߴ�
		*/
		Mat_(const Size3 &size_) : Mat(size_) {}
	};
	/**
	@brief MatCommaInitializer_ ������
	��Ϊ������������ʵ��
	Mat mat = (Mat_(3, 3) <<
		-1, -1, -1,
		-1,  9, -1,
		-1, -1, -1);
	*/
	class MatCommaInitializer_
	{
	public:
		explicit MatCommaInitializer_() {}
		MatCommaInitializer_(const Mat_& m) {
			head = m.mat_();
			it = head;
			row = m.rows();
			col = m.cols();
			channel = m.channels();
		}
		template<typename Tp_>
		MatCommaInitializer_ operator , (Tp_ v);
		int rows()const { return row; }
		int cols()const { return col; }
		int channels()const { return channel; }
		double * matrix()const { return head; }
	private:
		int row;
		int col;
		int channel;
		double *it;
		double *head;
	}; 
	template<typename Tp_>
	inline MatCommaInitializer_ MatCommaInitializer_::operator , (Tp_ v)
	{
		if (this->it == this->head + row*col*channel) {
			fprintf(stderr, errinfo[ERR_INFO_MEMOUT]);
			throw MatCommaInitializer_();
		}
		*this->it = double(v);
		++this->it;
		return *this;
	}

	template<typename Tp_>
	static MatCommaInitializer_ operator << (const Mat_& m, Tp_ val)
	{
		MatCommaInitializer_ commaInitializer(m);
		return (commaInitializer, val);
	}
	
	void check(int row, int col, int depth = 1);
	/**
	@brief �������������
	*/
	void Srandom();

	/**
	@brief ���ؾ�������ֵ
	@param src ����
	@param isAbs �Ƿ�ȡ����ֵ
	*/
	double Max(const Matrix &src, bool isAbs = false);
	/**
	@brief ���ؾ������Сֵ
	@param src ����
	@param isAbs �Ƿ�ȡ����ֵ
	*/
	double Min(const Matrix &src, bool isAbs = false);
	/**
	@brief ���ؾ��������ʽ
	@param src ����
	*/
	double det(const Matrix &src);
	/**
	@brief ���ؾ���ļ�
	@param src ����
	*/
	double trace(const Matrix &src);
	/**
	@brief ���ؾ����Ӧ����������ʽ
	@param src ����
	@param x ������
	@param y ������
	*/
	double cof(const Matrix &src, int x, int y);
	/**
	@brief ���������
	@param min ��Сֵ
	@param max ���ֵ
	@param isdouble �Ƿ����������
	*/
	double getRandData(int min, int max, bool isdouble = false);
	/**
	@brief ���ؾ�����
	@param src ����
	@param num ������
	*/
	double mNorm(const Matrix& src, int num = 1);
	/**
	@brief ���ؾ���ľ���
	@param a ����
	@param b ����
	@param num ������
	*/
	double mDistance(const Matrix& a, const Matrix& b, int num = 2);
	/**
	@brief ��������ľ���Ԫ��
	@param src ����
	*/
	double mRandSample(const Matrix &src);
	/**
	@brief ���ؽ�����ת����1*point.size()*1�ľ���
	@param point ����
	*/
	const Matrix VectoMat(std::vector<double> &point);
	/**
	@brief ���ؽ�����ת����point.size()*point[0].size()*1�ľ���
	@param points ����
	*/
	const Matrix VectoMat(std::vector<std::vector<double>> &points);
	/**
	@brief ���ؽ�����ת����һά����
	@param src ����
	*/
	std::vector<double> MattoVec(const Matrix &src);
	/**
	@brief ���ؽ�����ת����rowά����
	@param src ����
	*/
	std::vector<std::vector<double>> MattoVecs(const Matrix &src);
	/**
	@brief �������ɵ�n*n*1��λ����
	@param n �����С
	*/
	const Matrix eye(int n);
	/**
	@brief ���ؾ���ĵ�channel��ͨ��
	@param src ����
	@param channel ͨ������
	*/
	const Matrix mSplit(const Matrix &src, int channel);
	/**
	@brief ���ذ������ͨ��������
	@param src �������
	@param dst �������ͨ�����ľ�������
	*/
	void mSplit(const Matrix &src, Matrix *dst);
	/**
	@brief ���ذ�ͨ���ϲ��ľ���
	@param src ��������
	@param channels ͨ����
	*/
	const Matrix mMerge(const Matrix *src, int channels);
	/**
	@brief ���ذ����������и�ľ���
	@param src ����
	@param Row_Start ��ȡ�г�ʼ����ֵ
	@param Row_End ��ȡ�н�������ֵ
	@param Col_Start ��ȡ�г�ʼ����ֵ
	@param Col_End ��ȡ�н�������ֵ
	*/
	const Matrix Block(const Matrix &src, int Row_Start, int Row_End, int Col_Start, int Col_End);
	/**
	@brief �����������Ԫ��n*n*1����
	@param n �����С
	@param low �½�
	@param top �Ͻ�
	@param isdouble �Ƿ����ɸ�����
	*/
	const Matrix mRand(int low, int top, int n, bool isdouble = false);
	/**
	@brief �����������Ԫ��row*col*channel����
	@param row ��������
	@param col ��������
	@param low �½�
	@param top �Ͻ�
	@param isdouble �Ƿ����ɸ�����
	*/
	const Matrix mRand(int low, int top, int row, int col, int channel = 1,bool isdouble = false);
	/**
	@brief ����Ԫ��Ϊ0��row*col*1����
	@param row ��������
	@param col ��������
	*/
	const Matrix zeros(int row, int col);
	/**
	@brief ����Ԫ��Ϊ0��row*col*channel����
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Matrix zeros(int row, int col, int channel);
	/**
	@brief ����Ԫ��Ϊ0��size����
	@param size �����С
	*/
	const Matrix zeros(Size size);
	/**
	@brief ����Ԫ��Ϊ0��size����
	@param size �����С
	*/
	const Matrix zeros(const Size3 & size);
	/**
	@brief ����Ԫ��Ϊv��row*col*channel����
	@param v ���Ԫ��
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Matrix value(double v, int row, int col, int channel = 1);
	/**
	@brief ����Ԫ��Ϊ1��row*col*1����
	@param row ��������
	@param col ��������
	@param v ���Ԫ��
	*/
	const Matrix ones(int row, int col);
	/**
	@brief ����Ԫ��Ϊ0��row*col*channel����
	@param row ��������
	@param col ��������
	@param channel ����ͨ����
	*/
	const Matrix ones(int row, int col, int channel);
	/**
	@brief ����Ԫ��Ϊ0��size����
	@param size �����С
	*/
	const Matrix ones(Size size);
	/**
	@brief ����������󣬾�����Ϊһά����
	@param src ����
	*/
	const Matrix reverse(const Matrix &src);
	/**
	@brief ��������row*col*channel�ľ��������ȡ����srcԪ�����
	@param src ����
	@param row ��������
	@param col ��������
	*/
	const Matrix mRandSample(const Matrix &src, int row, int col, int channel = 1);
	/**
	@brief ���������ȡnum�ξ���src���л�����ɵľ���
	@param src ����
	@param rc ��ȡ��ʽ
	@param num ��ȡ����
	*/
	const Matrix mRandSample(const Matrix &src, X_Y_Z rc, int num = 1);
	/**
	@brief ���ش�low��top�ȷֳɵ�1*len�ľ���
	@param low �½�
	@param top �Ͻ�
	@param len �ȷָ���
	*/
	const Matrix linspace(int low, int top, int len);
	/**
	@brief ���ش�low��top�ȷֳɵ�1*len�ľ���
	@param low �½�
	@param top �Ͻ�
	@param len �ȷָ���
	*/
	const Matrix linspace(double low, double top, int len);
	/**
	@brief ���ؾ���İ������
	@param src ����
	*/
	const Matrix adj(const Matrix &src);
	/**
	@brief ���ؾ���������
	@param src ����
	*/
	const Matrix inv(const Matrix &src);
	/**
	@brief ���ؾ����α�����
	@param src ����
	@param dire α�����ļ��㷽ʽ
	*/
	const Matrix pinv(const Matrix &src, direction dire = LEFT);
	/**
	@brief ���ؾ����ת�þ���
	@param src ����
	*/
	const Matrix tran(const Matrix &src);
	/**
	@brief ���ؾ���ľ���ֵ����
	@param src ����
	*/
	const Matrix mAbs(const Matrix &src);
	/**
	@brief ����angle��2*2����ת����
	@param angle �Ƕ�
	*/
	const Matrix Rotate(double angle);
	/**
	@brief ���ؾ���num����
	@param src ����
	@param num ����
	*/
	const Matrix POW(const Matrix &src, int num);
	/**
	@brief ���ؾ���ȡ��
	@param src ����
	*/
	const Matrix mOpp(const Matrix &src);
	/**
	@brief ���ؾ����л���֮��
	@param src ����
	@param rc ��͵ķ���
	*/
	const Matrix mSum(const Matrix &src, X_Y_Z rc);
	/**
	@brief ���ؾ���Ԫ��ȡָ��
	@param src ����
	*/
	const Matrix mExp(const Matrix &src);
	/**
	@brief ���ؾ���Ԫ��ȡ����
	@param src ����
	*/
	const Matrix mLog(const Matrix &src);
	/**
	@brief ���ؾ���Ԫ��ȡ����
	@param src ����
	*/
	const Matrix mSqrt(const Matrix &src);
	/**
	@brief ���ؾ���Ԫ��ȡnum����
	@param src ����
	@param num ����
	*/
	const Matrix mPow(const Matrix &src, int num);
	/**
	@brief ���ؾ���val/src��Ԫ�س�
	@param src ����
	@param val ����
	*/
	const Matrix Divi(const Matrix &src, double val, direction dire = RIGHT);
	/**
	@brief ���ؾ������
	@param a ��������
	@param b ������
	@param dire ������ʽ
	*/
	const Matrix Divi(const Matrix &a, const Matrix &b, direction dire = RIGHT);
	/**
	@brief ���ؾ���Ԫ�ضԳ�
	@param a ����
	@param b ����
	*/
	const Matrix Mult(const Matrix &a, const Matrix &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮������ֵ
	@param a �Ƚ�ֵ
	@param b �ȽϾ���
	*/
	const Matrix mMax(double a, const Matrix &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮������ֵ
	@param a �ȽϾ���
	@param b �ȽϾ���
	*/
	const Matrix mMax(const Matrix &a, const Matrix &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮�����Сֵ
	@param a �Ƚ�ֵ
	@param b �ȽϾ���
	*/
	const Matrix mMin(double a, const Matrix &b);
	/**
	@brief ���ؾ���Ԫ��ȡa��b֮�����Сֵ
	@param a �ȽϾ���
	@param b �ȽϾ���
	*/
	const Matrix mMin(const Matrix &a, const Matrix &b);	
	/**
	@brief mCalSize �������������ŵı߽�
	���ؾ����С
	@param src ���������
	@param kern �����
	@param anchor ���ض�Ӧ���������
	anchorĬ��ΪPoint(-1,-1), ���ض�Ӧ���������
	@param strides ��������
	@param top �������伸��
	@param bottom �������伸��
	@param left �������伸��
	@param right �������伸��
	*/
	Size3 mCalSize(const Matrix &src, const Matrix &kern, Point & anchor, Size strides, int &top, int &bottom, int &left, int &right);
	/**
	@brief ���ذ�boundary�ֽ����ľ���
	���ؾ����С������������С
	@param src �������
	@param boundary �ֽ�ֵ
	@param lower С��boundary��lower���
	@param upper ����boundary��upper���
	@param boundary2upper ������Ԫ�ص���boundaryʱ
	Ϊ1��upper, Ϊ-1��lower, Ϊ0������
	*/
	const Matrix mThreshold(const Matrix &src, double boundary, double lower, double upper, int boundary2upper = 1);
	/**
	@brief ���ر߽�����ľ���
	@param src �������
	@param top �������伸��
	@param bottom �������伸��
	@param left �������伸��
	@param right �������伸��
	@param borderType �߽��������ƵĲ�ֵ����
	@param value ������ֵ����ֵ
	**/
	const Matrix copyMakeBorder(const Matrix &src, int top, int bottom, int left, int right, BorderTypes borderType = BORDER_CONSTANT, const int value = 0);
	/**
	@brief ���ؾ���2ά������
	���ؾ����СΪ(input.row/strides_x, input.col/strides_y, 1)
	@param input �������
	@param kern �����
	@param anchor ����Ԫ�ض�Ӧ����˵�λ��
	�Ծ���˵����Ͻ�Ϊ(0,0)��, Ĭ��(-1,-1)Ϊ����
	@param strides �������� 
	Size.heiΪx��,Size.widΪy��
	@param is_copy_border �Ƿ�Ҫ��չ�߽�
	*/
	const Matrix Filter2D(const Mat & input, const Mat & kern, Point anchor = Point(-1, -1), const Size & strides = Size(1, 1), bool is_copy_border = true);
	/**
	@brief �����а��������
	@param row ��
	@param col ��
	*/
	template<typename T>
	void showMatrix(const T *, int row, int col);
}

#endif //__MATRIX_H__
