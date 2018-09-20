#include "matrix.h"
#include "threadpool.h"
#include "jobinterface.h"
#include <random>
#include <ctime>
#include <stdexcept>
#include <iostream>
#include <algorithm>

ThreadPool *Matrix::threadPool_ = nullptr;

Matrix::Matrix(int nRows, int nCols) : nRows_(nRows), nCols_(nCols)
{
    if(nRows_ <= 0 || nCols_ <= 0) throw std::invalid_argument("String or column count has to be positive");
    const int size = nRows_ * nCols_;
    data_ = new double[size];
    for(int i = 0; i < size; ++i){
        data_[i] = 0;
    }
}

Matrix::Matrix(const Matrix& matrix) : nRows_(matrix.getRowCount()), nCols_(matrix.getColCount())
{
    const int size = nRows_ * nCols_;
    data_ = new double[size];
    for(int i = 0; i < size; ++i){
        data_[i] = matrix.data_[i];
    }
}

Matrix& Matrix::operator =(const Matrix& rVal){
    delete[] data_;
    nRows_ = rVal.getRowCount();
    nCols_ = rVal.getColCount();
    const int size = nRows_ * nCols_;
    data_ = new double[size];
    for(int i = 0; i < size; ++i){
        data_[i] = rVal.data_[i];
    }
    return *this;
}

double& Matrix::operator ()(int rowNo, int colNo){
    if(rowNo < 0 || colNo < 0 || rowNo >= nRows_ || colNo >= nCols_) throw std::invalid_argument("Wrong row or column position");
    return data_[rowNo * nCols_ + colNo];
}

double Matrix::operator ()(int rowNo, int colNo) const{
    if(rowNo < 0 || colNo < 0 || rowNo >= nRows_ || colNo >= nCols_) throw std::invalid_argument("Wrong row or column position");
    return data_[rowNo * nCols_ + colNo];
}

bool Matrix::operator==(const Matrix& rVal) const{
    if(rVal.getColCount() != nCols_ || rVal.getRowCount() != nCols_) return false;
    const int size = nCols_ * nRows_;
    for(int i = 0; i < size; ++i){
        if(data_[i] != rVal.data_[i]) return false;
    }
    return true;
}

void Matrix::setRandom(double minVal, double maxVal, bool setIntValues){
    if(minVal > maxVal) throw std::invalid_argument("maxVal has to be greater than minVal");

    const int size = nCols_ * nRows_;
    if(setIntValues){
        for(int i = 0; i < size; ++i){
            data_[i] = int(minVal + rand() * double(maxVal - minVal) / RAND_MAX);
        }
    }
    else {
        for(int i = 0; i < size; ++i){
            data_[i] = minVal + rand() * double(maxVal - minVal) / RAND_MAX;
        }
    }
}

void Matrix::init(int numThreads){
    if(numThreads <= 0) throw std::invalid_argument("Number of threads has to be positive");
    srand(time(0));

    if(threadPool_) delete threadPool_;
    threadPool_ = new ThreadPool(numThreads);
}

void Matrix::deleteThreadPool(){
    if(threadPool_ == nullptr) return;
    delete threadPool_;
    threadPool_ = nullptr;
}

Matrix::~Matrix(){
    delete[] data_;
}


Matrix Matrix::operator +(const Matrix& rval) const{
    if(getColCount() != rval.getColCount() || getRowCount() != rval.getRowCount())
        throw std::invalid_argument("Matrixes has to be the same size");
    Matrix mat = *this;
    const int size = nRows_ * nCols_;
    for(int i = 0; i < size; ++i){
        mat.data_[i] += rval.data_[i];
    }
    return mat;
}

Matrix Matrix::operator -(const Matrix& rval) const{
    if(getColCount() != rval.getColCount() || getRowCount() != rval.getRowCount())
        throw std::invalid_argument("Matrixes has to be the same size");
    Matrix mat = *this;
    const int size = nRows_ * nCols_;
    for(int i = 0; i < size; ++i){
        mat.data_[i] -= rval.data_[i];
    }
    return mat;
}

Matrix& operator-(Matrix& mat){
    const int size = mat.nRows_ * mat.nCols_;
    for(int i = 0; i < size; ++i){
        mat.data_[i] *= -1;
    }
    return mat;
}

Matrix Matrix::getTransposed() const{
    Matrix result(getColCount(), getRowCount());
    const int size = nRows_ * nCols_;
    for(int i = 0; i < size; ++i){
        result.data_[i] = data_[(i * getColCount()) % size + i * getColCount() / size];
    }
    return result;
}


void swapCol(Matrix& matrix, int col1No, int col2No, int rowFrom){
    for(int i = rowFrom; i < matrix.getRowCount(); ++i){
        std::swap(matrix(i, col1No), matrix(i, col2No));
    }
}

void swapRow(Matrix& matrix, int row1No, int row2No, int colFrom){
    for(int i = colFrom; i < matrix.getColCount(); ++i){
        std::swap(matrix(row1No, i), matrix(row2No, i));
    }
}


//MULTIPLICATION
class JobMultiplication : public JobInterface{
    static const double __restrict_arr *data1_; // matAns = mat1 * mat2
    static const double __restrict_arr *data2_;
    static double __restrict_arr *dataRes_;
    static bool calcByRow_;
    const std::vector<int> indexes_;
    static int rowCount1_;
    static int colCount1_;
    static int rowCount2_;
    static int colCount2_;
    static int finishedJobs_;
    static int countJobs_;
    static pthread_mutex_t counter_;
public:
    static pthread_mutex_t finished;
    static pthread_cond_t finishedCond;

    JobMultiplication(const std::vector<int> indexes) : indexes_(indexes) { ; }

    void working(){
        if(calcByRow_){
            for(const auto curRow : indexes_){
                for(int i = 0; i < colCount2_; ++i){
                    for(int j = 0; j < rowCount2_; ++j){
                         dataRes_[curRow * colCount2_ + i] += data1_[curRow * colCount1_ + j] * data2_[j * colCount2_ + i];
                    }
                }
            }
        }
        else{
            for(const auto curCol : indexes_){
                for(int i = 0; i < rowCount1_; ++i){
                    for(int j = 0; j < colCount1_; ++j){
                        dataRes_[i * colCount2_ + curCol] += data1_[i * colCount1_ + j] * data2_[j * colCount2_ + curCol];
                    }
                }
            }
        }

        pthread_mutex_lock(&counter_);
        ++finishedJobs_;
        if(finishedJobs_ == countJobs_){
            pthread_mutex_lock(&finished);
            pthread_cond_signal(&finishedCond);
            pthread_mutex_unlock(&finished);
        }
        pthread_mutex_unlock(&counter_);
    }
    static void init(double *dataRes, const double *data1, const double *data2, int rowCount1,
                     int colCount1, int rowCount2, int colCount2, int countJobs, bool calcByRow){
        dataRes_ = dataRes;
        data1_ = data1;
        data2_ = data2;
        rowCount1_ = rowCount1;
        colCount1_ = colCount1;
        rowCount2_ = rowCount2;
        colCount2_ = colCount2;
        finishedJobs_ = 0;
        countJobs_ = countJobs;
        calcByRow_ = calcByRow;
        pthread_mutex_destroy(&finished);
        pthread_mutex_destroy(&counter_);
        pthread_cond_destroy(&finishedCond);
        pthread_mutex_init(&finished, NULL);
        pthread_mutex_init(&counter_, NULL);
        pthread_cond_init(&finishedCond, NULL);
    }
};

double *JobMultiplication::dataRes_ = nullptr;
const double *JobMultiplication::data1_ = nullptr;
const double *JobMultiplication::data2_ = nullptr;
int JobMultiplication::rowCount1_ = 0;
int JobMultiplication::colCount1_ = 0;
int JobMultiplication::rowCount2_ = 0;
int JobMultiplication::colCount2_ = 0;
int JobMultiplication::finishedJobs_ = 0;
int JobMultiplication::countJobs_ = 0;
bool JobMultiplication::calcByRow_ = false;

pthread_mutex_t JobMultiplication::finished = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t JobMultiplication::counter_ = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t JobMultiplication::finishedCond = PTHREAD_COND_INITIALIZER;

Matrix Matrix::operator *(const Matrix& rval) const
{
    if(threadPool_ == nullptr) throw std::logic_error("No thread pool initialization");
    if(getColCount() != rval.getRowCount()) throw std::invalid_argument("Size of matrixes is not appropriate for multiplication");
    
    Matrix result(getRowCount(), rval.getColCount());
    
    bool calcByRow = true;
    if(rval.getColCount() < getRowCount()) calcByRow = false;


    std::vector<std::vector<int>> indexes(threadPool_->getThreadCount());

    if(calcByRow){
        for(int i = 0; i < result.getRowCount(); ++i){
            indexes[i % indexes.size()].push_back(i);
        }
    }
    else{
        for(int i = 0; i < result.getColCount(); ++i){
            indexes[i % indexes.size()].push_back(i);
        }
    }

    JobMultiplication::init(result.data_, data_, rval.data_, getRowCount(), getColCount(), rval.getRowCount(), rval.getColCount(), indexes.size(), calcByRow);
    
    pthread_mutex_lock(&JobMultiplication::finished);
    for(size_t i = 0; i < indexes.size(); ++i){
        JobMultiplication *job = new JobMultiplication(indexes[i]);
        threadPool_->assignJob(job);
    }
    pthread_cond_wait(&JobMultiplication::finishedCond, &JobMultiplication::finished);
    pthread_mutex_unlock(&JobMultiplication::finished);

    return result;
}

//END_MULTIPLICATION

//DETERMINANT

class JobDet : public JobInterface{
    const int number_; //number from 0 to nThreads - 1
    static double __restrict_arr *data_;

    static int matrixSize_; // size of matrix : [matrixSize_ * matrixSize_]
    static int countDone_;
    static int nThreads_;
    static pthread_mutex_t mutexCounter_;
public:

    static pthread_cond_t condNextStep;
    static pthread_mutex_t mutexNextStep;
    static int curStep;

    JobDet(int number) : number_(number) { ; }

    void working(){
        int count = 0;
        for(int curRow = number_ + curStep + 1; curRow < matrixSize_; curRow += nThreads_, ++count){
            double k = data_[matrixSize_ * curRow + curStep] / data_[matrixSize_ * curStep + curStep];
            for(int i = curStep; i < matrixSize_; ++i){
                data_[matrixSize_ * curRow + i] -= k * data_[matrixSize_ * curStep + i];
            }
        }

        pthread_mutex_lock(&mutexCounter_);
        countDone_ += count;
        if(countDone_ == matrixSize_ - curStep - 1){
            pthread_mutex_lock(&mutexNextStep);
            countDone_ = 0;
            pthread_cond_signal(&condNextStep);
            pthread_mutex_unlock(&mutexNextStep);
        }
        pthread_mutex_unlock(&mutexCounter_);
    }
    static void init(double *data, int matrixSize, int nThreads){
        data_ = data;
        nThreads_ = nThreads;
        matrixSize_ = matrixSize;
        curStep = 0;
        countDone_ = 0;
        pthread_mutex_destroy(&mutexNextStep);
        pthread_mutex_destroy(&mutexCounter_);
        pthread_cond_destroy(&condNextStep);
        pthread_mutex_init(&mutexNextStep, NULL);
        pthread_mutex_init(&mutexCounter_, NULL);
        pthread_cond_init(&condNextStep, NULL);
    }
};

double *JobDet::data_ = nullptr;
int JobDet::curStep = 0;
int JobDet::matrixSize_ = 0;
int JobDet::countDone_ = 0;
int JobDet::nThreads_ = 0;
pthread_mutex_t JobDet::mutexCounter_ = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t JobDet::mutexNextStep = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t JobDet::condNextStep = PTHREAD_COND_INITIALIZER;

double Matrix::getDeterminant(double precision) const{
    if(threadPool_ == nullptr) throw std::logic_error("No thread pool initialization");
    if(getColCount() != getRowCount()) throw std::invalid_argument("Not square matrix");
    if(precision < 0) throw std::invalid_argument("Presision must be positive");
    
    const int size = nCols_ * nRows_;
    double *temp = new double[size];
    for(int i = 0; i < size; ++i){
        temp[i] = data_[i];
    }
    JobDet::init(temp, getColCount(), threadPool_->getThreadCount());
    double res = 1;
    for(int i = 0; i < getColCount() - 1; ++i){
        pthread_mutex_lock(&JobDet::mutexNextStep);
        int nullCount = 0;
        int curRowSwap = i + 1;
        while(abs(temp[i * getColCount() + i]) <= precision){
            ++nullCount;
            if(nullCount == getColCount() - i){
                pthread_mutex_unlock(&JobDet::mutexNextStep);
                delete[] temp;
                return 0;
            }
            for(int j = i; j < getColCount(); ++j){
                std::swap(temp[getColCount() * i + j], temp[getColCount() * curRowSwap + j]);
            }
            res *= -1;
            ++curRowSwap;
        }
        for(int j = 0; j < threadPool_->getThreadCount(); ++j){
            JobDet *job = new JobDet(j);
            threadPool_->assignJob(job);
        }

        pthread_cond_wait(&JobDet::condNextStep, &JobDet::mutexNextStep);
        ++JobDet::curStep;
        pthread_mutex_unlock(&JobDet::mutexNextStep);
    }

    for(int i = 0; i < getColCount(); ++i){
        res *= temp[i * getColCount() + i];
    }

    delete[] temp;
    return res;
}

//END_DETERMINANT


//INVERSE

class JobInverse : public JobInterface{
    const int number_;
    static double* __restrict_arr initialData_;
    static double* __restrict_arr resData_;
    static int matrixSize_;
    static int countDone_;
    static int nThreads_;
    static pthread_mutex_t mutexCounter_;
public:

    static pthread_cond_t condNextStep;
    static pthread_mutex_t mutexNextStep;
    static int curStep_;
    JobInverse(int number) : number_(number){ ; }
    void working(){
        int count  = 0;
        for(int curRow = number_; curRow < matrixSize_; curRow += nThreads_, ++count){
            if(curRow == curStep_) continue;
            double k = initialData_[matrixSize_ * curRow + curStep_] / initialData_[matrixSize_ * curStep_ + curStep_];
            for(int i = curStep_; i < matrixSize_; ++i){
                initialData_[matrixSize_ * curRow + i] -= k * initialData_[matrixSize_ * curStep_ + i];
            }
            for(int i = 0; i < matrixSize_; ++i){
                resData_[matrixSize_ * curRow + i] -= k * resData_[matrixSize_ * curStep_ + i];
            }

        }

        pthread_mutex_lock(&mutexCounter_);
        countDone_ += count;

        if(countDone_ == matrixSize_){
            pthread_mutex_lock(&mutexNextStep);
            countDone_ = 0;
            pthread_cond_signal(&condNextStep);
            pthread_mutex_unlock(&mutexNextStep);
        }
        pthread_mutex_unlock(&mutexCounter_);
    }
    static void init(double *initialData, double *resData, int matrixSize, int nThreads){
        initialData_ = initialData;
        resData_ = resData;
        nThreads_ = nThreads;
        matrixSize_ = matrixSize;
        curStep_ = 0;
        countDone_ = 0;
        pthread_mutex_destroy(&mutexNextStep);
        pthread_mutex_destroy(&mutexCounter_);
        pthread_cond_destroy(&condNextStep);
        pthread_mutex_init(&mutexNextStep, NULL);
        pthread_mutex_init(&mutexCounter_, NULL);
        pthread_cond_init(&condNextStep, NULL);
    }
};

double *JobInverse::initialData_ = nullptr;
double *JobInverse::resData_ = nullptr;
int JobInverse::curStep_ = 0;
int JobInverse::matrixSize_ = 0;
int JobInverse::countDone_ = 0;
int JobInverse::nThreads_ = 0;
pthread_mutex_t JobInverse::mutexCounter_ = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t JobInverse::mutexNextStep = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t JobInverse::condNextStep = PTHREAD_COND_INITIALIZER;


Matrix Matrix::getInverse(double precision) const{
    if(threadPool_ == nullptr) throw std::logic_error("No thread pool initialization");
    if(getColCount() != getRowCount()) throw std::invalid_argument("Not square matrix");
    if(precision < 0) throw std::invalid_argument("Presision must be positive");

    Matrix inversed(getColCount(), getRowCount());

    const int size = nRows_ * nCols_;
    for(int i = 0; i < inversed.getColCount(); ++i){
        inversed.data_[i * getColCount() + i] = 1;
    }

    double *temp = new double[size];
    for(int i = 0; i < size; ++i){
        temp[i] = data_[i];
    }
    JobInverse::init(temp, inversed.data_, getColCount(), threadPool_->getThreadCount());
    for(int i = 0; i < getColCount(); ++i){
        pthread_mutex_lock(&JobInverse::mutexNextStep);
        int nullCount = 0;
        int curRowSwap = i + 1;
        while(std::abs(temp[i * getColCount() + i]) <= precision){
            ++nullCount;
            if(nullCount == getColCount()){
                pthread_mutex_unlock(&JobInverse::mutexNextStep);
                throw std::runtime_error("Null determinant");
                delete[] temp;
                return Matrix(getRowCount(), getColCount());
            }
            for(int j = i; j < getColCount(); ++j){
                std::swap(temp[getColCount() * i + j], temp[getColCount() * curRowSwap + j]);
            }
            for(int j = 0; j < getColCount(); ++j){
                std::swap(inversed.data_[getColCount() * i + j], inversed.data_[getColCount() * curRowSwap + j]);
            }
            ++curRowSwap;
        }
        for(int j = 0; j < threadPool_->getThreadCount(); ++j){
            JobInverse *job = new JobInverse(j);
            threadPool_->assignJob(job);
        }
        pthread_cond_wait(&JobInverse::condNextStep, &JobInverse::mutexNextStep);
        ++JobInverse::curStep_;
        pthread_mutex_unlock(&JobInverse::mutexNextStep);
    }

    for(int i = 0; i < getColCount(); ++i){
        double k = temp[i * getColCount() + i];
        for(int j = 0; j < getColCount(); ++j){
            inversed.data_[i * getColCount() + j] /= k;
        }
    }

    delete[] temp;
    return inversed;

}
//END_INVERSE


//RANK

class JobRank : public JobInterface{
    const int number_;
    static double* __restrict_arr data_;
    static int nRows_;
    static int nCols_;
    static int countDone_;
    static int nThreads_;
    static pthread_mutex_t mutexCounter_;
public:

    static pthread_cond_t condNextStep;
    static pthread_mutex_t mutexNextStep;
    static int curStep;
    JobRank(int number) : number_(number) { ; }
    void working(){
        int count = 0;
        for(int curRow = number_ + curStep + 1; curRow < nRows_; curRow += nThreads_, ++count){
            double k = data_[nCols_ * curRow + curStep] / data_[nCols_ * curStep + curStep];
            for(int i = curStep; i < nCols_; ++i){
                data_[nCols_ * curRow + i] -= k * data_[nCols_ * curStep + i];
            }
        }
        pthread_mutex_lock(&mutexCounter_);
        countDone_ += count;
        if(countDone_ == nRows_ - curStep - 1){
            pthread_mutex_lock(&mutexNextStep);
            countDone_ = 0;
            pthread_cond_signal(&condNextStep);

            pthread_mutex_unlock(&mutexNextStep);
        }
        pthread_mutex_unlock(&mutexCounter_);
    }
    static void init(double *data, int nCols, int nRows, int nThreads){
        data_ = data;
        nThreads_ = nThreads;
        nCols_ = nCols;
        nRows_ = nRows;
        curStep = 0;
        countDone_ = 0;
        pthread_mutex_destroy(&mutexNextStep);
        pthread_mutex_destroy(&mutexCounter_);
        pthread_cond_destroy(&condNextStep);
        pthread_mutex_init(&mutexNextStep, NULL);
        pthread_mutex_init(&mutexCounter_, NULL);
        pthread_cond_init(&condNextStep, NULL);
    }
};

double *JobRank::data_ = nullptr;
int JobRank::curStep = 0;
int JobRank::nCols_ = 0;
int JobRank::nRows_ = 0;
int JobRank::countDone_ = 0;
int JobRank::nThreads_ = 0;
pthread_mutex_t JobRank::mutexCounter_ = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t JobRank::mutexNextStep = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t JobRank::condNextStep = PTHREAD_COND_INITIALIZER;

int Matrix::getRank(double precision) const{
    if(threadPool_ == nullptr) throw std::logic_error("No thread pool initialization");
    if(precision < 0) throw std::invalid_argument("Presision must be positive");
    Matrix temp = *this;
    if(getColCount() < getRowCount()){
        temp = temp.getTransposed();
    }
    JobRank::init(temp.data_, temp.getColCount(), temp.getRowCount(), threadPool_->getThreadCount());
    int rank = 0;
    for(int i = 0; i < temp.getRowCount(); ++i){

        if(std::abs(temp.data_[i * temp.getColCount() + i]) <= precision){
            int j = i + 1;
            while(std::abs(temp.data_[i * temp.getColCount() + j]) <= precision && j < temp.getColCount()) ++j;
            if(j == temp.getColCount()){
                j = i + 1;
                while(std::abs(temp.data_[j * temp.getColCount() + i]) <= precision && j < temp.getRowCount()) ++j;
                if(j == temp.getRowCount()) continue;
                else swapRow(temp, j, i, i);
            }
            else swapCol(temp, i, j, i);
        }
        ++rank;
        if(i == temp.getRowCount() - 1) break;
        pthread_mutex_lock(&JobRank::mutexNextStep);
        for(int j = 0; j < threadPool_->getThreadCount(); ++j){
            JobRank *job = new JobRank(j);
            threadPool_->assignJob(job);
        }
        pthread_cond_wait(&JobRank::condNextStep, &JobRank::mutexNextStep);
        ++JobRank::curStep;
        pthread_mutex_unlock(&JobRank::mutexNextStep);
    }
    return rank;
}

//END_RANK


//SOLVESYSTEM

class JobSystem : public JobInterface{
    const int number_;
    static double* __restrict_arr data_;
    static int nRows_;
    static int nCols_;
    static int countDone_;
    static int nThreads_;

    static pthread_mutex_t mutexCounter_;
public:
    static pthread_cond_t condNextStep;
    static pthread_mutex_t mutexNextStep;
    static int curStep;
    JobSystem(int number) : number_(number){ ; }
    void working(){
        int count  = 0;
        for(int curRow = number_; curRow < nRows_; curRow += nThreads_, ++count){
            if(curRow == curStep) continue;
            double k = data_[nCols_ * curRow + curStep] / data_[nCols_ * curStep + curStep];
            for(int i = curStep; i < nCols_; ++i){
                data_[nCols_ * curRow + i] -= k * data_[nCols_ * curStep + i];
            }
        }
        pthread_mutex_lock(&mutexCounter_);
        countDone_ += count;
        if(countDone_ == nRows_){
            pthread_mutex_lock(&mutexNextStep);
            countDone_ = 0;
            pthread_cond_signal(&condNextStep);
            pthread_mutex_unlock(&mutexNextStep);
        }
        pthread_mutex_unlock(&mutexCounter_);
    }
    static void init(double *data, int nRows, int nCols, int nThreads){
        data_ = data;
        nThreads_ = nThreads;
        nCols_ = nCols;
        nRows_ = nRows;
        curStep = 0;
        countDone_ = 0;
        pthread_mutex_destroy(&mutexNextStep);
        pthread_mutex_destroy(&mutexCounter_);
        pthread_cond_destroy(&condNextStep);
        pthread_mutex_init(&mutexNextStep, NULL);
        pthread_mutex_init(&mutexCounter_, NULL);
        pthread_cond_init(&condNextStep, NULL);
    }
};

double *JobSystem::data_ = nullptr;

int JobSystem::curStep = 0;
int JobSystem::nRows_ = 0;
int JobSystem::nCols_ = 0;
int JobSystem::countDone_ = 0;
int JobSystem::nThreads_ = 0;
pthread_mutex_t JobSystem::mutexCounter_ = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t JobSystem::mutexNextStep = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t JobSystem::condNextStep = PTHREAD_COND_INITIALIZER;


Matrix solveHomogeneousLinearSystem(const Matrix &a, double precision){
    if(precision < 0) throw std::invalid_argument("Presision must be positive");
    const int rank = a.getRank(precision);
    if(rank == a.getColCount()) return Matrix(a.getColCount(), 1);
    std::vector<int> x(a.getColCount());
    for(size_t i = 0; i < x.size(); ++i) x[i] = i;

    Matrix temp = a;
    JobSystem::init(temp.data_, temp.getRowCount(), temp.getColCount(), a.threadPool_->getThreadCount());

    for(int i = 0; i < rank; ++i){
        if(std::abs(temp.data_[i * a.getColCount() + i]) <= precision){
            int j = i + 1;
            while(std::abs(temp.data_[j * temp.getColCount() + i]) <= precision && j < temp.getRowCount()) ++j;
            if(j == temp.getRowCount()){
                j = i + 1;
                while(std::abs(temp.data_[i * temp.getColCount() + j]) <= precision && j < temp.getColCount()) ++j;
                if(j == temp.getColCount()) {
                    for(int curRow = i + 1; curRow < temp.getRowCount() && temp(i, i) == 0; ++curRow){
                        for(int curCol = i + 1; curCol < temp.getColCount() && temp(i, i) == 0; ++curCol){
                            if(temp(curRow, curCol) != 0){
                                swapRow(temp, i, curRow, i);
                            }
                        }
                    }
                    --i;
                    continue;
                }
                else{
                    swapCol(temp, i, j, 0);
                    std::swap(x[j], x[i]);
                }
            }
            else swapRow(temp, i, j, i);
        }

        pthread_mutex_lock(&JobSystem::mutexNextStep);
        for(int j = 0; j < a.threadPool_->getThreadCount(); ++j){
            JobSystem *job = new JobSystem(j);
            a.threadPool_->assignJob(job);
        }

        pthread_cond_wait(&JobSystem::condNextStep, &JobSystem::mutexNextStep);

        ++JobSystem::curStep;
        pthread_mutex_unlock(&JobSystem::mutexNextStep);
    }

    for(int i = 0; i < rank; ++i){
        double k = temp.data_[i * temp.getColCount() + i];
        for(int j = rank; j < temp.getColCount(); ++j){
            temp.data_[i * temp.getColCount() + j] /= k;
        }
    }
    Matrix result(a.getColCount(), a.getColCount() - rank);
    for(int curRow = 0; curRow < rank; ++curRow){
        for(int curCol = 0; curCol < result.getColCount(); ++curCol){
            result.data_[curRow * result.getColCount() + curCol] = -temp.data_[curRow * temp.getColCount() + rank + curCol];
        }
    }
    for(int i = 0; i < result.getColCount(); ++i){
        result.data_[(i + rank) * result.getColCount() + i] = 1;
    }
    for(size_t i = 0; i < x.size(); ++i){
        for(size_t j = i + 1; j < x.size(); ++j){
            if(x[i] > x[j]){
                std::swap(x[i], x[j]);
                swapRow(result, i, j, 0);
            }
        }
    }

    return result;
}

//END_SOLVESYSTEM
