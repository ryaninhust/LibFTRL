#include <iostream>
#include <cstring>
#include <stdexcept>

#include "ftrl.h"

#include <fenv.h>


struct Option
{
    shared_ptr<Parameter> param;
    FtrlInt verbose, solver;
    string data_path, test_path;
};

string basename(string path)
{
    const char *ptr = strrchr(&*path.begin(), '/');
    if(!ptr)
        ptr = path.c_str();
    else
        ptr++;
    return string(ptr);
}

bool is_numerical(char *str)
{
    int c = 0;
    while(*str != '\0')
    {
        if(isdigit(*str))
            c++;
        str++;
    }
    return c > 0;
}

string train_help()
{
    return string(
    "usage: train [options] training_set_file test_set_file\n"
    "\n"
    "options:\n"
    "-s <solver>: set solver type (default 1)\n"
	"	 0 -- AdaGrad framework\n"
	"	 1 -- FTRL framework\n"
	"	 2 -- RDA framework\n"
    "-a <alpha>: set initial learning rate\n"
    "-b <beta>: set shrinking base for learning rate schedule\n"
    "-l1 <lambda_1>: set regularization coefficient on l1 regularizer (default 0.1)\n"
    "-l2 <lambda_2>: set regularization coefficient on l2 regularizer (default 0.1)\n"
    "-t <iter>: set number of iterations (default 20)\n"
    "-p <path>: set path to test set\n"
    "-c <threads>: set number of cores\n"
    "--norm: Apply instance-wise normlization."
    "--no-auc: disable auc\n"
	"--in-memory: keep data in memroy\n"
    "--auto-stop: stop at the iteration that achieves the best validation loss (must be used with -p)\n"
    );
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option option;
    option.verbose = 1;
    option.param = make_shared<Parameter>();
    int i = 0;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-s") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify solver type\
                                        after -s");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-s should be followed by a number");
            option.solver = atoi(argv[i]);
        }
        else if(args[i].compare("-l1") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify l1 regularization\
                                        coefficient after -l1");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-l1 should be followed by a number");
            option.param->l1 = atof(argv[i]);
        }
        else if(args[i].compare("-l2") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify l2\
                                        regularization coefficient\
                                        after -l2");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-l2 should be followed by a number");
            option.param->l2 = atof(argv[i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            option.param->nr_pass = atoi(argv[i]);
        }
        else if(args[i].compare("-a") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->alpha = atof(argv[i]);
        }
        else if(args[i].compare("-b") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->beta = atof(argv[i]);
        }
        else if(args[i].compare("-c") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->nr_threads = atof(argv[i]);
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;

            option.test_path = string(args[i]);
        }
        else if(args[i].compare("--norm") == 0)
        {
            option.param->normalized = true;
        }
        else if(args[i].compare("--verbose") == 0)
        {
            option.param->verbose = true;
        }
        else if(args[i].compare("--freq") == 0)
        {
            option.param->freq = true;
        }
        else if(args[i].compare("--auto-stop") == 0)
        {
            option.param->auto_stop = true;
        }
        else if(args[i].compare("--no-auc") == 0)
        {
            option.param->no_auc = true;
        }
        else if(args[i].compare("--in-memory") == 0)
        {
            option.param->in_memory = true;
        }
        else
        {
            break;
        }
    }

    if(i >= argc)
        throw invalid_argument("training data not specified");
    option.data_path = string(args[i++]);

    return option;
}

int main(int argc, char *argv[])
{
    try
    {
        Option option = parse_option(argc, argv);
        omp_set_num_threads(option.param->nr_threads);

        shared_ptr<FtrlData> data = make_shared<FtrlData>(option.data_path);
        shared_ptr<FtrlData> test_data = make_shared<FtrlData>(option.test_path);
		data->split_chunks();
        cout << "Tr_data: ";
        data->print_data_info();

        if (!test_data->file_name.empty()) {
            test_data->split_chunks();
            cout << "Va_data: ";
            test_data->print_data_info();
        }

        FtrlProblem prob(data, test_data, option.param);
        prob.initialize();
        if (option.solver == 1) {
            cout << "Solver Type: FTRL" << endl;
            prob.solve();
        }
        else if (option.solver == 2) {
            cout << "Solver Type: RDA" << endl;
            prob.solve_rda();
        }
        else {
            cout << "Solver Type: AdaGrad" << endl;
            prob.solve_adagrad();
        }
        string model_path = basename(option.data_path) + ".model";
        prob.save_model(model_path.c_str());
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}
