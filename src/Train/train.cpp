#include "train.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char **argv) {
    if (argc < 4) {
        cout << "Not enough arguments. 4 arguments required" << endl;
        throw 1;
    }
    if (argc > 4) {
        cout << "Too many arguments. Only 4 arguments allowed" << endl;
        throw 1;
    }
    const string data_path = argv[1];
    const string params_path = argv[2];
    const std::filesystem::path model_path = argv[3];
    LogisticRegression model(params_path);
    model.load_data(data_path, true);
    model.train();
    model.save_model(model_path);
    return 0;
}
