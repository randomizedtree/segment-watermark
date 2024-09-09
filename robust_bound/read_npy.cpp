#include "npy.hpp"
#include "read_npy.h"
#include <vector>
#include <string>

/* read from a npy file which contains a double 2d ndarray, then return a 2d array in cpp with the same value*/
void read_2d(const std::string path, ndarray2& res){
    npy::npy_data d = npy::read_npy<double>(path);

    auto data = d.data;
    auto shape = d.shape;

    for(int r = 0; r < shape[0]; r++){
        res.push_back(vector<double>());
        for(int c = 0; c < shape[1]; c++){
            res[r].push_back(data[r*shape[1] + c]);
        }
    }
}

void read_4d(const std::string path, ndarray4& res){
    npy::npy_data d = npy::read_npy<float>(path);

    auto data = d.data;
    auto shape = d.shape;

    for(int t = 0; t < shape[0]; t++){
        res.push_back(ndarray3());
        for(int s = 0; s < shape[1]; s++){
            res[t].push_back(ndarray2());
            for(int r = 0; r < shape[2]; r++){
                res[t][s].push_back(vector<double>());
                for(int c = 0; c < shape[3]; c++){
                    int index = t * shape[1] * shape[2] * shape[3] + s * shape[2] * shape[3] + r * shape[3] + c;
                    res[t][s][r].push_back(data[index]);
                }
            }
        }
    }
}
