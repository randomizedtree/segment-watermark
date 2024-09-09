import numpy as np
import hashlib
import struct
from typing import List
import pickle

# dangerous
import subprocess

def get_rand(seed):
    s=seed.to_bytes(16, 'little', signed=False)

    t=hashlib.sha256(s).digest()

    inx =struct.unpack('I',t[0:4])[0]
    return inx

def get_mapping(KEY: int, freq: List[float], p: int, 
                path_freq_shuffled: str = './freq_shuffled.txt', 
                path_res_len: str = './res_len.txt'):
    # shuffle with KEY
    indices = [i for i in range(len(freq))]
    np.random.seed(get_rand(KEY))
    np.random.shuffle(indices)

    freq_shuffled = np.array(freq)[indices]

    with open(path_freq_shuffled, 'w') as fw:
        for i in freq_shuffled:
            fw.write(str(i))
            fw.write('\n')

    # FIXME: dangerous
    commands = ['./dp_balance', path_freq_shuffled, path_res_len, str(p)]
    process = subprocess.Popen(commands)
    process.wait()
    
    with open(path_res_len, 'r') as fr:
        lines = fr.readlines()
        res_len = [int(s) for s in lines]

    # construct mapping
    mapping = {}
    base = 0
    seg_idx = 0
    for l in res_len:
        for i in range(l):
            mapping[indices[base + i]] = seg_idx
        seg_idx += 1
        base += l

    return mapping

def gen_mapping(p: int, path: str, save_path: str, KEY: int = 426371835) -> dict:
    """
    Read token-freq array from a pkl file and save the mapping (a dict) into a pickle file
    Token with 0 occurrence will be patched to 1

    :param p: the number of buckets, or n in RS
    """

    with open(path, 'rb') as f:
        freq = pickle.load(f)

    # patch token with 0 freq
    mask = np.array(freq) == 0
    freq_patched = freq + mask

    assert sum(np.array(freq_patched) == 0) == 0

    print("* Start generation of balance hash mapping")

    mapping = get_mapping(KEY, freq=freq_patched, p=p)

    print("* Finished generation of balance hash mapping")

    with open(save_path, 'wb') as fw:
        pickle.dump(mapping, fw)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--gf_segment_num', type=int, required=True, help="Number of buckets to divide the tokens")
    parser.add_argument('--path', required=True, help="path to the freq array pickle file")
    parser.add_argument('--save', required=True, help="path for saving the resulting mapping")
    parser.add_argument('-k', '--key', type=int, required=False, help="secret key")

    args = parser.parse_args()

    if not args.key:
        m = gen_mapping(p=args.gf_segment_num, path=args.path, save_path=args.save)
    else:
        m = gen_mapping(p=args.gf_segment_num, path=args.path, save_path=args.save, KEY=args.key)

    # print(m)