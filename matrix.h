#ifndef MATRIX_H
#define MATRIX_H

class ThreadPool;

class Matrix
{
    int nRows_;
    int nCols_;
    double *data_;
    static ThreadPool *threadPool_;
public:    
    Matrix(int nRows, int nCols);
    Matrix(const Matrix&);
    Matrix& operator=(const Matrix&);

    int getColCount() const{
        return nCols_;
    }
    int getRowCount() const{
        return nRows_;
    }

    double& operator()(int rowNo, int colNo);
    double operator()(int rowNo, int colNo) const;
    void setRandom(double minVal, double maxVal, bool setIntValues = false);

    bool operator==(const Matrix&) const;

    Matrix operator+(const Matrix&) const;
    Matrix operator-(const Matrix&) const;


    friend Matrix& operator-(Matrix&);

    Matrix operator*(const Matrix&) const;
    double getDeterminant(double precision = 0.0001) const;
    Matrix getInverse(double precision = 0.0001) const;
    Matrix getTransposed() const;
    int getRank(double precision = 0.00001) const;

    static void init(int numThreads);
    static void deleteThreadPool();
    friend Matrix solveHomogeneousLinearSystem(const Matrix& a, double precision);
    ~Matrix();
};

Matrix solveHomogeneousLinearSystem(const Matrix& a, double precision = 0.0001);

#endif // MATRIX_H
