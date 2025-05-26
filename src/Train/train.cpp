#include "train.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "Not enough arguments. 2 arguments required" << endl;
        throw 1;
    }
    if (argc > 3) {
        cout << "Too many arguments. Only 1 argument allowed" << endl;
        throw 1;
    }
    const string data_path = argv[1];
    const string params_path = argv[2];
    LogisticRegression model(params_path);
    model.load_data(data_path, true);
    model.train();
    model.save_model("./output/logistic_regression.model");
    return 0;
}
