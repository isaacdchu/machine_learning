#include "evaluate.h"

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
        cout << "Too many arguments. Only 2 arguments allowed" << endl;
        throw 1;
    }
    const string data_path = argv[1];
    const string model_file = argv[2];
    LogisticRegression model(model_file);
    model.load_data(data_path, true);
    model.evaluate();
    return 0;
}
