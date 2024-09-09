import json
import os
import argparse
from typing import List
from generalizedReedSolomon.generalizedreedsolo import Generalized_Reed_Solomon

def get_segment_tokens(segs: List[int], n: int):
    res = [0 for i in range(n)]
    for i in segs:
        res[i] += 1

    return res

def get_correct_payload_score(score: List[List[int]], payload: List[int]):
    res = []
    for i in range(len(score)):
        res.append(score[i][payload[i]])

    return res

def get_payload(payload: int, gf_segments_num: int, segments_num: int, segment_bit: int) -> List[int]:
    """
    Given a payload in number, return a payload in segs

    :param gf_segments_num: n
    :param segments_num: k
    :param segment_bit: m
    """
    mask = 2 ** segment_bit - 1
    segments = [
        (payload >> (segment_bit * i)) & mask for i in range(segments_num)
    ]

    # 2. encode this 5 symbols with RS
    rs = Generalized_Reed_Solomon(
        field_size=2,
        message_length=gf_segments_num,
        payload_length=segments_num,
        symbol_size=segment_bit,
        p_factor=1
    )

    gf_segments = [int(i) for i in rs.encode(segments)]
    return gf_segments

def toks_assignment_for_setting(n: int, k: int, m: int, payload: int, score_path: str, save_path: str) -> None:

    payload = get_payload(payload, gf_segments_num=n, segments_num=k, segment_bit=m)
    segment_tokens = []
    correct_payload_scores = []

    with open(score_path, 'r') as f:
        ls = f.readlines()

        for l in ls:
            l = json.loads(l)
            score = l['score']
            tokens_seg = l['tokens_segment']
            sgmt_tks_one_smpl = get_segment_tokens(tokens_seg, n=n)
            crct_pld_scr_one_smpl = get_correct_payload_score(score, payload)

            correct_payload_scores.append(crct_pld_scr_one_smpl)
            segment_tokens.append(sgmt_tks_one_smpl)

    with open(save_path, 'a') as fw:
        for i in range(len(segment_tokens)):
            fw.write(str(segment_tokens[i]))
            fw.write(';')
            fw.write(str(correct_payload_scores[i]))
            fw.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", required=True, help="Path to scores.jsonl file to analyze. We assume the payload are consistent inside this file." +
                        " If not, please separate the content of scores.jsonl (one file per line) and run for each individually")
    parser.add_argument("--save_path", required=True, help="Path to save the preprocessing results")
    parser.add_argument("--n", required=True, help="segment number after encoding")
    parser.add_argument("--k", required=True, help="segment number before encoding")
    parser.add_argument("--m", required=True, help="bit num of one segment")
    parser.add_argument("--payload", required=True, help="Payload used to generate the content")

    args = parser.parse_args()

    toks_assignment_for_setting(
        n=int(args.n),
        k=int(args.k),
        m=int(args.m),
        payload=int(args.payload),
        score_path=args.score_path,
        save_path=args.save_path,
    )
