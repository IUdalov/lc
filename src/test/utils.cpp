#include "utils.h"

#include <sstream>
#include <string>
#include <fstream>

#include <debug.h>

using namespace lc;
int readCSVFile(const std::string& path, Problem& data) {
    FUNCTION_LOG;
    std::ifstream csvFile(path);
    std::string line;
    while(std::getline(csvFile, line)) {
        std::cout << line << std::endl;
        std::string temp = "";
        lc::Node node;
        for(size_t i = 0; i < line.size(); i++) {
            switch(line[i]) {
                case ',':
                    node.push_back(atof(temp.c_str()));
                    temp = "";
                    break;
                case ';':
                    data.push_back(std::make_pair(node, atoi(temp.c_str())));
                    temp = "";
                    break;
                case ' ':
                    break;
                default:
                    temp.push_back(line[i]);
                    break;
            }
        }
    }
    return 0;
}