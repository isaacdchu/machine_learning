#include "predict.h"

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
    const string model_file = argv[2];
    const string raw_contains_label = argv[3];
    const bool contains_label = (raw_contains_label == "true");
    LogisticRegression model(model_file);
    model.load_data(data_path, false);
    model.predict();
    return 0;
}
