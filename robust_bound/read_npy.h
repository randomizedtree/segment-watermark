#ifndef READNPY
#define READNPY

#include <string>
#include <vector>

#define i128 __int128_t
#define vector std::vector
#define ndarray2 vector<vector<double>>
#define ndarray3 vector<vector<vector<double>>>
#define ndarray4 vector<vector<vector<vector<double>>>>

void read_2d(const std::string path, ndarray2& res);
void read_4d(const std::string path, ndarray4& res);
#endif
