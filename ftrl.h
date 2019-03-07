#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <cstring>

#include "omp.h"

using namespace std;


typedef double FtrlFloat;
typedef long FtrlInt;
typedef long long FtrlLong;

FtrlLong const chunk_size = 3000000000;

class Node {
public:
    FtrlLong idx;
    FtrlFloat  val;
    Node(){};
    Node(FtrlLong idx, FtrlFloat val): idx(idx), val(val){};
};

class Parameter {

public:
    FtrlFloat l1, l2, alpha, beta;
    FtrlInt nr_pass, nr_threads;
    bool normalized, verbose, freq, auto_stop, no_auc, in_memory;
    Parameter():l1(0.1), l2(0.1), alpha(0.1), beta(1), normalized(false),verbose(true), freq(true), auto_stop(false), no_auc(false), in_memory(false), nr_threads(1){};
};

class FtrlChunk {
public:
    FtrlLong l, nnz;
    FtrlInt chunk_id;
    string file_name;

    vector<Node> nodes;
    vector<FtrlInt> nnzs;
    vector<FtrlFloat> labels;
    vector<FtrlFloat> R;


    void read();
    void write();
    void clear();

    FtrlChunk(string data_name, FtrlInt chunk_id);
};

class FtrlData {
public:
    string file_name;
    FtrlLong l, n;
    FtrlInt nr_chunk;

    vector<FtrlChunk> chunks;

    FtrlData(string file_name): file_name(file_name), l(0), n(0), nr_chunk(0) {};
    void print_data_info();
    void split_chunks();
    void write_meta();
};

class FtrlProblem {
public:
    shared_ptr<FtrlData> data;
    shared_ptr<FtrlData> test_data;
    shared_ptr<Parameter> param;
    FtrlProblem() {};
    FtrlProblem(shared_ptr<FtrlData> &data, shared_ptr<FtrlData> &test_data, shared_ptr<Parameter> &param)
        :data(data), test_data(test_data), param(param) {};


    vector<FtrlFloat> w, z, n, f;
    FtrlInt t;
    FtrlFloat tr_loss, va_loss, va_auc, fun_val, gnorm, reg;
    FtrlFloat start_time;

    void initialize();
    void solve();
    void solve_adagrad();
    void solve_rda();
    void print_epoch_info();
    void print_header_info();
    void save_model(string model_path);
    FtrlLong load_model(string model_path);
    void fun();
    void validate();
};

