# Provably Robust Multi-bit Watermarking for AI-generated Text

Demo code for paper submission "Provably Robust Multi-bit Watermarking for AI-generated Text".

## Download data

Before running any of the following program, please first **download necessary data files** from [Data file link](https://drive.google.com/file/d/1CIX-VAz24L03jTHaRk3kEcZ73_jGQQha/view?usp=drive_link)

This zip file contains:
- datasets files (`C4.jsonl`, `essays.jsonl`, `OpenGen.jsonl`)
- token frequency pkl files (`token_freq_llama.pkl`, `token_freq_guanaco.pkl`, `token_freq_falcon.pkl`)
- precomputed `HYPER.npy` and `F.npy`

Please `unzip data_share.zip` after download and use the corresponding file path in the following commands.

## Example usage
This section introduces an example usage of our watermark scheme given message length as 32 bits.

Suppose the above data files are unzipped and placed at root folder of this project.

For method used in paper, firstly we need to generate the frequency mapping using code under `balance_hash/` folder:
```shell
cd balance_hash
make
python3 helper.py -p 6 --path ../token_freq_llama.pkl --save './map_freq.pkl'
```
Then move to the root folder of the project (`cd ..`) and run the following to run watermarked generation with standard method described in paper:

```shell
python3 main.py --model_name llama-7b \
    --prompt_path "./OpenGen.jsonl" --nsamples 100 --batch_size 1 \
    --method rsbh --temperature 1.0 --seeding hash --ngram 1 --method_detect same --scoring_method none \
    --payload_mode random --payload_max 4294967296 --gamma 0.5 --delta 6.0 --max_gen_len 200 \
    --bh_map_load_path "./balance_hash/map_freq.pkl" \
    --gf_segments_num 6 --segments_num 4 --segment_bit 8 \
    --output_dir "./output"
```

This will run both embedding and extracting, which will 
- save generation results and extraction results in `./output/results.jsonl` and `./output/scores.jsonl` respectively
- save statistics of match rate, average decoding time, etc. in `output_log.txt`

## Details

### RS code selection

The `rs_search.py` script searches for optimal RS scheme under message length of 12, 16, 20, 24, 28, 32 bits

```shell
python3 rs_search.py
```

This will give you output in format (n, k, m), where n is the number of encoded segments, k is the number of segments before encoding, m is the bit length of each segment

### Generation of mapping

```shell
cd balance_hash
make
python3 helper.py -p <n> --path <path_to_token_freq.pkl> --save 'path/to/map_freq.pkl'
```

where `n` is the number of encoded segments

### Text generation with watermarks & detection

```shell
python3 main.py --model_name <model_name> \
    --prompt_path "path/to/dataset" --nsamples <sample_num> --batch_size 1 \
    --method rsbh --temperature 1.0 --seeding hash --ngram 1 --method_detect same --scoring_method none \
    --payload_mode random --payload_max <payload_max> --gamma 0.5 --delta 6.0 --max_gen_len <T> \
    --bh_map_load_path "path/to/map_freq.pkl" \
    --gf_segments_num <n> --segments_num <k> --segment_bit <m> \
    --output_dir "path/to/folder_saving_results(2 jsonl files)"
```
Here 
- `--payload_max <payload_max>` should be the number of $2 ^ b$, where $b$ is the message length
- `--output_dir` specify the folder to save `results.jsonl` and `scores.jsonl` which save the generated text and extraction results. If this option is set to a folder which already contains a `results.jsonl` with N samples, `main.py` will skip the generation phase for the first N samples
- `<n>`, `<k>`, `<m>` are the RS scheme parameters: number of encoded segments, number of original segments, bit number for each segment
- `<T>` maximum text generation length

### Robust bound computation

To compute robust bound of watermarked text, first we need to have scores.jsonl already generated.

Then use generate the segment tokens assignment results using `tok_assignment.py`:

```shell
python3 tok_assignment.py --score_path path/to/scores.jsonl --save_path path/to/tok_assign.txt --n <n> --k <k> --m <m> --payload <payload>
```
where we assume all samples in this `scores.jsonl` file shares the same message specified in `--payload <payload>`. If not, you can divide them into multiple files and run them with their corresponding message value

- `--n <n>`: segment number after encoding
- `--k <k>`: segment number before encoding
- `--m <m>`: bit number of each segment
- `--save_path`: specify the path to save the results, which will be used later

After generation of `tok_assign.txt` which contains the number of tokens assigned to each segment and the number of green tokens assigned to each segment, we can compile and run the program under `/robust_bound` to calculate the robust bound of each sample.

Compilation:

```shell
cd robust_bound
mkdir build
cd build
cmake ..
make
```
Then run the program with:
```shell
./robust_bound path/to/tok_assign.txt path/to/HYPER.npy path/to/F.npy path/to/save.txt
```

The robust bound of each samples will be saved in `path/to/save.txt` with each line representing the robust bound of one sample.

## Acknowledgements

We developed the code with reference to [Three Bricks to Consolidate Watermarks for LLMs](https://github.com/facebookresearch/three_bricks).

The Reed Solomon code implementation is adapted from [Generalized Reed Solomon code](https://github.com/raeudigerRaeffi/generalizedReedSolomon).

We use `libnpy` to read `.npy` file. The code `robust_bound/npy.hpp` is adapted from [libnpy](https://github.com/llohse/libnpy/blob/master/include/npy.hpp).

