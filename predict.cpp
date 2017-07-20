#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <cstdlib>

#include "ftrl.h"

using namespace std;

struct Option
{
    string test_path, model_path, output_path;
};

string predict_help()
{
    return string(
"usage: predict test_file model_file output_file\n");
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;

    if(argc != 4)
        throw invalid_argument("cannot parse argument");

    option.test_path = string(args[1]);
    option.model_path = string(args[2]);
    option.output_path = string(args[3]);

    return option;
}

void predict(string test_path, string model_path, string output_path)
{
    FtrlProblem prob;
    FtrlLong n = prob.load_model(model_path);
    ofstream f_out(output_path);

    shared_ptr<FtrlData> test_data = make_shared<FtrlData>(test_path);
    test_data->split_chunks();
    cout << "Te_data: ";
    test_data->print_data_info();

    FtrlInt nr_chunk = test_data->nr_chunk;
    FtrlFloat local_va_loss = 0.0;

    for (FtrlInt chunk_id = 0; chunk_id < nr_chunk; chunk_id++) {

        FtrlChunk chunk = test_data->chunks[chunk_id];
        chunk.read();

        for (FtrlInt i = 0; i < chunk.l; i++) {

            FtrlFloat y = chunk.labels[i], wTx = 0;

            for (FtrlInt s = chunk.nnzs[i]; s < chunk.nnzs[i+1]; s++) {
                Node x = chunk.nodes[s];
                FtrlInt idx = x.idx;
                if (idx > n) {
                    continue;
                }
                FtrlFloat val = x.val;
                wTx += prob.w[idx]*val;
            }

            FtrlFloat exp_m;

            if (wTx*y > 0) {
                exp_m = exp(-y*wTx);
                local_va_loss += log(1+exp_m);
            }
            else {
                exp_m = exp(y*wTx);
                local_va_loss += -y*wTx+log(1+exp_m); 
            }
            f_out << 1/(1+exp(-wTx)) << "\n";
        }
        chunk.clear();
    }
    local_va_loss = local_va_loss / test_data->l;
    cout << "logloss = " << fixed << setprecision(5) << local_va_loss << endl;
}

int main(int argc, char **argv)
{
    Option option;
    try
    {
        option = parse_option(argc, argv);
    }
    catch(invalid_argument const &e)
    {
        cout << e.what() << endl;
        return 1;
    }

    predict(option.test_path, option.model_path, option.output_path);
}
