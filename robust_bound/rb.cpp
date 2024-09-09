#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include "read_npy.h"

#define eta 40

typedef double ArrayT[2][2*eta+1][2*eta+1];

/* Implementation of combination */
i128 nCr(int n, int r)
{
    if(r == 0) return 1;
    if(r > n/2) return nCr(n, n - r); 

    i128 res = 1; 
    for (int k = 1; k <= r; ++k)
    {
        res *= n - k + 1;
        res /= k;
    }
    return res;
}

double bin_prob(int n, int k, int B, double*** BIN){
    /* prob of n trials, success k trials, with 1/B prob success */
    if(k > n) return 0;
    if(BIN[n][k][B] != 0 ) return BIN[n][k][B];

    double p = 1.0 / (double)B;
    BIN[n][k][B] = nCr(n, k) * pow(p, k) * pow(1-p, n-k);
    return BIN[n][k][B];
}

/* compute pp[kk+2][2][2*eta+1][2*eta+1] */
void p_computation(
    int kk,
    vector<int>& z_array, 
    vector<int>& x_array,
    ndarray4& HYPER,
    ndarray4& F,
    double (*pp)[2][2*eta+1][2*eta+1], 
    double*** BIN)
{
    // k = 2
    int z = z_array[2];
    int x = x_array[2];
    int zk_sum = 0;
    for(int j = 1; j < 3; j++) { zk_sum += z_array[j]; }
    
    const auto& sub_HYPER = HYPER[zk_sum][z];
    const auto& sub_F     = F[z][x];
    const auto& sub_F1    = F[z_array[1]][x_array[1]];
    for(int a = 0; a < 2*eta+1; a++){
        for(int b = 0; b < 2*eta+1; b++){
            double sum1 = 0;
            double sum2 = 0;
            for(int ak = 0; ak < a+1; ak++){
                double p_ak = bin_prob(a, ak, 2, BIN);
                for(int bk = 0; bk < b+1; bk++){
                    double p_bk = sub_HYPER[b][bk] * p_ak;
                    sum1 += p_bk * (1-sub_F1[a-ak][b-bk])*(1-sub_F[ak][bk]);
                    sum2 += p_bk * ((1-2*sub_F1[a-ak][b-bk]) * sub_F[ak][bk] + sub_F1[a-ak][b-bk]);
                }
            }
            pp[2][0][a][b] = sum1;
            pp[2][1][a][b] = sum2;
        }
    }


    // k > 2
    for(int k = 3; k < kk+1; k++){
        z = z_array[k];
        x = x_array[k];
        int zk_sum = 0;
        for(int j = 1; j < k+1; j++) { zk_sum += z_array[j]; }
        
        const auto& sub_HYPER = HYPER[zk_sum][z];
        const auto& sub_F     = F[z][x];
        for(int a = 0; a < 2*eta+1; a++){
            for(int b = 0; b < 2*eta+1; b++){
                double sum1 = 0;
                double sum2 = 0;
                for(int ak = 0; ak < a+1; ak++){
                    double p_ak = bin_prob(a, ak, k, BIN);
                    for(int bk = 0; bk < b+1; bk++){
                        double p_bk = sub_HYPER[b][bk] * p_ak;
                        sum1 += p_bk * pp[k-1][0][a-ak][b-bk]*(1-sub_F[ak][bk]);
                        sum2 += p_bk * ((pp[k-1][0][a-ak][b-bk]-pp[k-1][1][a-ak][b-bk]) * sub_F[ak][bk] + pp[k-1][1][a-ak][b-bk]);
                    }
                }
                pp[k][0][a][b] = sum1;
                pp[k][1][a][b] = sum2;
            }
        }
    }
}

/* binary search for robust bound */
int binary_search(
    vector<int> z_array,
    vector<int> x_array, 
    ndarray4& HYPER,
    ndarray4& F,
    double*** BIN)
{
    int kk = x_array.size() - 1;
    auto zk = *std::max_element(z_array.begin(), z_array.end());
    auto x_const = *std::max_element(x_array.begin(), x_array.end());
    auto f_index_max = std::max(80, zk);

    // skip when any z_array > 60
    bool has_greater_than_60 = std::any_of(z_array.begin(), z_array.end(), [](int x) {
        return x > 60;
    });
    if(has_greater_than_60) return 0;

    // init pp
    ArrayT* pp = new ArrayT[kk+2];

    p_computation(kk, z_array, x_array, HYPER, F, pp, BIN); 

    int left = 0;
    int right = eta;

    while(left <= right){
        int mid = left + (right - left) / 2;
        if((1-pp[kk][0][2*mid][2*mid]-pp[kk][1][2*mid][2*mid]) < 0.001){
            left = mid + 1;
        }
        else{
            right = mid - 1;
        }
    }
    return right >= 0? right : 0;
}


void parseTokAssgnF(const std::string& path, vector<vector<int>>& z_arrays, vector<vector<int>>& x_arrays){
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }

    std::string line;
    int count = 0;
    while(getline(file, line)){
        std::istringstream iss(line);
        std::string v1, v2;
        getline(iss, v1, ';');
        getline(iss, v2);

        // get rid of []
        v1 = v1.substr(1, v1.length() - 2);
        v2 = v2.substr(1, v2.length() - 2);

        std::istringstream iss_v1(v1);
        std::istringstream iss_v2(v2);
        std::string tmp;
        // add 0 in front
        z_arrays.push_back({});
        x_arrays.push_back({});
        z_arrays[count].push_back(0);
        x_arrays[count].push_back(0);
        while(getline(iss_v1, tmp, ',')){
            z_arrays[count].push_back(std::stoi(tmp));
        }
        while(getline(iss_v2, tmp, ',')){
            x_arrays[count].push_back(std::stoi(tmp));
        }
        count ++;
    }
}

int main(int argc, char* argv[]){
    /*
        argv[1]: path to tok_assignment file
        argv[2]: path to HYPER npy file
        argv[3]: path to F npy file
        argv[4]: path to save robust bounds
    */
    if(argc < 5){
        std::cerr << "Usage: ./robust_bound path/to/tok_assign.txt path/to/HYPER.npy path/to/F.npy path/to/save.txt" << std::endl;
        return 1;
    }

    // load HYPER and F
    ndarray4 HYPER, F;
    read_4d(argv[2], HYPER);
    read_4d(argv[3], F);
    // read vector of z_array and vector of x_array
    vector<vector<int>> z_arrays, x_arrays;
    parseTokAssgnF(argv[1], z_arrays, x_arrays);
    
    // init BIN
    double*** BIN = new double**[500];
    for(int i = 0; i < 500; i++){
        BIN[i] = new double*[500];
        for(int j = 0; j < 500; j++){
            BIN[i][j] = new double[13];
        }
    } 

    // number of samples
    int n = z_arrays.size();

    // robust bound for each sample
    vector<int> robust_bounds;

    for(int i = 0; i < n; i++){
        std::cout << "Start Iteration " << i << std::endl;
        // Record time
        auto start = std::chrono::high_resolution_clock::now();

        robust_bounds.push_back(binary_search(z_arrays[i], x_arrays[i], HYPER, F, BIN));

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();
        // Calculate duration
        std::chrono::duration<double> duration = end - start;
        std::cout << "Iteration " << i << " took " << duration.count() << " seconds" << std::endl;
    }

    // write to save file
    std::ofstream os(argv[4]);
    if (!os.is_open()){
        std::cerr << "Unable to open write file\n";
        return 1;
    }
    for(const auto& i : robust_bounds){
        os << i << std::endl;
    }
    os.close();

    return 0;
}