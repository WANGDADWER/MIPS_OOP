#ifndef MIPS_H_INCLUDED
#define MIPS_H_INCLUDED
#include <stdlib.h>
#include <iostream> // cin, cout
#include <fstream> // fscanf, fopen, ofstream
#include <Eigen/Dense>

using namespace Eigen;

#include "header.h"
//Define a base class MIPS with the purpose of solving MIPS problems
//Other algorithms (CEOs, LSH) will inherit this class and extend it
//for their own needs
class MIPS{

protected:
    // Input
    int PARAM_DATA_N = 0; // Number of points (rows) of X
    int PARAM_QUERY_Q = 0; // Number of rows (queries) of Q
    int PARAM_DATA_D = 0; // Number of dimensions

    // Internal params
    bool PARAM_INTERNAL_SAVE_OUTPUT = false; // save the results

    // MIPS
    int PARAM_MIPS_TOP_K = 0; // TopK largest entries from Xq
    int PARAM_MIPS_TOP_B = 0; // number of points to compute dot products

    // Input
    //Matrix<float, Dynamic, Dynamic, ColMajor> MATRIX_X; // matrix X
    MatrixXf MATRIX_X;
    //extern Matrix<float, Dynamic, Dynamic, ColMajor> MATRIX_Q; // matrix Q
    MatrixXf MATRIX_Q;


public:
    
    MIPS():PARAM_DATA_N(0),PARAM_QUERY_Q(0),PARAM_DATA_D(0),
    PARAM_MIPS_TOP_K(0),PARAM_MIPS_TOP_B(0),PARAM_INTERNAL_SAVE_OUTPUT(false){}
    MIPS(int N, int Q, int D, int K, int B, bool save);

    void read_X_from_file(const char *path);  //read the file of matrix X MATRIX_X

    void read_Q_from_file(const char *path);  //read the file of matrix Q MATRIX_Q
    
    //Load X and Q from numpy array
    void read_X_from_np(const Eigen::Ref<const MatrixXf> &); //read X from numpy array
    
    void read_Q_from_np(const Eigen::Ref<const MatrixXf> &);
    
    // compute distance and extract top-K
    void extract_TopK_MIPS(const Ref<VectorXf>&, const Ref<VectorXi>&, int, Ref<VectorXi>);
    
    Eigen::MatrixXf get_X(); //get X matrix
    Eigen::MatrixXf get_Q(); //get Q matrix
    
    
    
};








#endif // MIPS_H_INCLUDED
