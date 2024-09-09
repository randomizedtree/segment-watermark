#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <string>

#define i128 __int128_t
#define INT128_MAX (((__int128_t(1) << 126) - 1) * 2 + 1)

void min_and_argmin(std::vector<i128>& arr, i128& min_value, int& min_index){
    /* get min value and its index from arr, save in min_value and min_index */
    min_value = INT128_MAX; // init with i128_max
    int n = arr.size();
    for(int i = 0; i < n; i++){
        if(arr[i] < min_value){
            min_value = arr[i];
            min_index = i;
        }
    }
}

std::vector<i128> prefix_sum(std::vector<i128>& arr){
    /* given an array, return the prefix sum array */
    int n = arr.size();
    std::vector<i128> res(n+1, 0);
    i128 running_sum = 0;
    res[0] = running_sum;

    for(int i = 0; i < n; i++){
        running_sum += arr[i];
        res[i+1] = running_sum;
    }

    return res;
}

__always_inline i128 D(int i, int j, std::vector<i128>& prefix_sum_freq){
    i128 interval_sum = prefix_sum_freq[j+1] - prefix_sum_freq[i];
    return interval_sum * interval_sum;
}

std::vector<int> dp_balance(std::vector<i128>& freq, int p){
    int n = freq.size();
    /* calculate prefix sum */
    auto prefix_sum_freq = prefix_sum(freq);

    /* init Q */
    i128** Q = new i128*[n];
    for(int i = 0; i < n; i++) { Q[i] = new i128[p]; }

    /* init best_index */
    int** best_index = new int*[n];
    for(int i = 0; i < n; i++) { best_index[i] = new int[p]; }

    /* deal with j = 0 */
    for(int i = 0; i < n - p + 1; i++){ Q[i][0] = D(n-i-1, n-1, prefix_sum_freq); }

    /* deal with j = 1 ~ p-1 */
    for(int j = 1; j < p; j++){
        std::cout << "* Start DP round " << j << " out of " << p << std::endl;
        // Record start time
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = j; i < n - p + j + 1; i++){
            std::vector<i128> candidates(i-j+1);
            for(int k = j-1; k < i; k++){
                candidates[k-(j-1)] = Q[k][j-1] + D(n-i-1, n-k-2, prefix_sum_freq);
            }

            /* select minimal one among i-j+1 candidates */
            i128 min_value; int min_index;
            min_and_argmin(candidates, min_value, min_index);

            /* best_index[i][j] stores which option Q[i][j] choose, 
               the value chosen is: 
                Q[best_index[i][j]][j-1] + D(N-i, N-best_index[i][j]-1)
            */
            Q[i][j] = min_value; best_index[i][j] = min_index + j - 1;
        }
        
        // Record end time
        auto end = std::chrono::high_resolution_clock::now();
        // Calculate duration
        std::chrono::duration<double> duration = end - start;
        // print time usage for round j
        std::cout << "Iteration " << j << " took " << duration.count() << " seconds" << std::endl;
    }

    int i = n - 1; int j = p - 1;
    std::vector<int> segment_indices {best_index[i][j]};
    while(j > 1){
        i = best_index[i][j];
        j--;
        segment_indices.push_back(best_index[i][j]);
    }

    std::vector<int> res_len { n - segment_indices[0] - 1};
    for(int i = 1; i < p-1; i++){
        res_len.push_back(segment_indices[i-1] - segment_indices[i]);
    }
    res_len.push_back(segment_indices[segment_indices.size() - 1] + 1);

    return res_len;
}

int main(int argc, char* argv[]){
    if(argc < 4){
        std::cout << "Usage: dp_balance <freq_shuffle_filename> <res_len_save_filename> <p>" << std::endl;
    }

    std::ifstream freq_shuffled_file(argv[1]);
    if (!freq_shuffled_file.is_open()) {              // Check if file opened successfully
        std::cerr << "Unable to open freq_shuffle file\n";
        return 1;
    }

    std::vector<i128> freq;
    std::string line;
    while(std::getline(freq_shuffled_file, line)){
        freq.push_back((i128)std::stoll(line));
    }

    freq_shuffled_file.close();

    auto res_len = dp_balance(freq, std::stoi(argv[3]));

    std::ofstream os(argv[2]);
    if (!os.is_open()){
        std::cerr << "Unable to open write file\n";

        for(const auto& num: res_len){
            std::cout << num << '\n';
        }
        return 1;
    }

    for (const auto& num : res_len) {
        os << num << '\n'; // Add a newline character after each number
    }    
    os.close();

    return 0;
}