import argparse
from typing import Dict, List
import os
import time
import json

import tqdm
import pandas as pd
import numpy as np
import random
import torch
from peft import PeftModel    
from transformers import AutoModelForCausalLM, AutoTokenizer

from wm import (WmGenerator, OpenaiGenerator, OpenaiDetector, 
                OpenaiDetectorZ, MarylandGenerator, MarylandDetector, MarylandDetectorZ,
                MarylandGeneratorE, MarylandDetectorE, 
                RSGenerator, RSDecoder, RSBHGenerator, RSBHDecoder,
                BCHGenerator, BCHDecoder, get_pvalue_segment_based)
from utils import (bool_inst, bitacc, load_prompts, load_prompts_by_index, load_results, load_res_payload)
from math import log2
import pickle

def get_args_parser():
    parser = argparse.ArgumentParser('Args', add_help=False)

    # model parameters
    parser.add_argument('--model_name', type=str)

    # prompts parameters
    parser.add_argument('--prompt_path', type=str, default="data/OpenGen.json")
    parser.add_argument('--prompt', type=str, nargs='+', default=None, 
                        help='prompt to use instead of prompt_path, can be a list')

    # generation parameters
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_gen_len', type=int, default=256)

    # watermark parameters
    parser.add_argument('--method', type=str, default='none', 
                        help='Choose among: none (no watermarking), openai (Aaronson et al.), maryland (Kirchenbauer et al.), marylandE' +
                        '(extension to maryland which supports possible message number > vocab size), rs (Segment assignment-based with ECC)' +
                        ', rsbh (Segment assignment-based with ECC & balanced segment assignment), bch (using BCH as ECC)')
    parser.add_argument('--method_detect', type=str, default='same',
                        help='Statistical test to detect watermark. By default using same.')

    parser.add_argument('--seeding', type=str, default='hash', 
                        help='seeding method for rng key generation as introduced in https://github.com/jwkirchenbauer/lm-watermarking')
    parser.add_argument('--ngram', type=int, default=1, 
                        help='watermark context width for rng key generation')
    parser.add_argument('--gamma', type=float, default=0.5, 
                        help='gamma for maryland: proportion of greenlist tokens')
    parser.add_argument('--delta', type=float, default=4.0, 
                        help='delta for maryland: bias to add to greenlist tokens')
    parser.add_argument('--hash_key', type=int, default=35317, 
                        help='hash key for rng key generation')
    parser.add_argument('--scoring_method', type=str, default='none', 
                        help='method for scoring. choose between: \
                        none (score every tokens), v1 (score token when wm context is unique), \
                        v2 (score token when {wm context + token} is unique')

    # ECC parameters
    parser.add_argument('--segments_num', type=int, default=8, required=False,
                        help='[Used only in RS & RSBH] number of message segments, or k in RS')
    parser.add_argument('--gf_segments_num', type=int, default=14, required=False,
                        help='[Used only in RS & RSBH] number of message segments after ECC, or n in RS')
    parser.add_argument('--segment_bit', type=int, default=4, required=False,
                        help='[Used only in RS & RSBH] number of bit inside one segment, or m in RS (GF(q^m))')
    parser.add_argument('--bh_map_load_path', required=False, default='./balance_hash/mapping/map.pkl',
                        help='[Used only in RSBH] path to the saved balance hash mapping, should be a pickle file. When this option is set,' +
                        ' program will load mapping directly from the file instead of generate it (i.e. freq_path, map_save_path will not be used in this case)')

    # multibit
    parser.add_argument('--payload', type=int, default=0, help='message')
    parser.add_argument('--payload_max', type=int, default=0, 
                        help='maximal message, must be inferior to the vocab size at the moment')
    parser.add_argument('--payload_mode', type=str, default='random', help='whether to randomize payload used in each paragraph. To use this option' +
                        'payload_max must be set!')

    # expe parameters
    parser.add_argument('--nsamples', type=int, default=None, 
                        help='number of samples to generate, if None, take all prompts')
    # Only support batch_size = 1 currently
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--do_eval', type=bool_inst, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--output_log_path', type=str, default="./output_log.txt")
    parser.add_argument('--split', type=int, default=None,
                        help='split the prompts in nsplits chunks and chooses the split-th chunk. \
                        Allows to run in parallel. \
                        If None, treat prompts as a whole')
    parser.add_argument('--nsplits', type=int, default=None,
                        help='number of splits to do. If None, treat prompts as a whole')

    # distributed parameters
    parser.add_argument('--ngpus', type=int, default=None)

    return parser

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # build model
    if args.model_name == "llama-7b":
        model_name = "huggyllama/llama-7b"
        adapters_name = None
    if args.model_name == "llama-chat-7b":
        model_name = "daryl149/llama-2-7b-chat-hf"
        adapters_name = None
    elif args.model_name == 'guanaco':
        model_name = "huggyllama/llama-7b"
        adapters_name = "timdettmers/guanaco-7b"
    elif args.model_name == 'falcon':
        model_name = 'tiiuae/falcon-7b'
        adapters_name = None
    # elif args.model_name == 'vicuna':
    #     model_name = 'lmsys/vicuna-7b-v1.5'
    #     adapters_name = None
    # elif args.model_name == "guanaco-13b":
    #     model_name = "huggyllama/llama-13b"
    #     adapters_name = 'timdettmers/guanaco-13b'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    args.ngpus = torch.cuda.device_count() if args.ngpus is None else args.ngpus
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory={i: '32000MB' for i in range(args.ngpus)},
        offload_folder="offload",
    )
    if adapters_name is not None:
        model = PeftModel.from_pretrained(model, adapters_name)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Using {args.ngpus}/{torch.cuda.device_count()} GPUs - {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated per GPU")

    # check payload mode: fixed payload vs. randomized payload
    if args.payload_mode == 'random':
        args.payload = random.randint(0, args.payload_max - 1)

    # build watermark generator
    if args.method == "none":
        generator = WmGenerator(model, tokenizer)
    elif args.method == "openai":
        generator = OpenaiGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, payload=args.payload)
    elif args.method == "maryland":
        generator = MarylandGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, payload=args.payload, gamma=args.gamma, delta=args.delta)
    elif args.method == "marylandE":
        generator = MarylandGeneratorE(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, payload=args.payload, gamma=args.gamma, delta=args.delta, payload_max=args.payload_max)
    elif args.method == "rs":
        generator = RSGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, payload=args.payload, gamma=args.gamma, delta=args.delta,
                                segments_num=args.segments_num, gf_segments_num=args.gf_segments_num, segment_bit=args.segment_bit)
    elif args.method == "rsbh":
        if args.bh_map_load_path:
            with open(args.bh_map_load_path, 'rb') as f:
                bh_mapping = pickle.load(f)
        else: 
            raise ValueError('token_freq.pkl path not provided in balanced segment assignment!')

        generator = RSBHGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, payload=args.payload, gamma=args.gamma, delta=args.delta,
                                segments_num=args.segments_num, gf_segments_num=args.gf_segments_num, segment_bit=args.segment_bit,
                                bh_mapping=bh_mapping)
    elif args.method == "bch":
        generator = BCHGenerator(model, tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, payload=args.payload, gamma=args.gamma, delta=args.delta, payload_max=args.payload_max)
    else:
        raise NotImplementedError("method {} not implemented".format(args.method))

    # load prompts
    if args.prompt is not None:
        prompts = args.prompt
        prompts = [{"instruction": prompt} for prompt in prompts]
    else:
        prompts = load_prompts(json_path=args.prompt_path, nsamples=args.nsamples)

    # do splits
    if args.split is not None:
        nprompts = len(prompts)
        left = nprompts * args.split // args.nsplits 
        right = nprompts * (args.split + 1) // args.nsplits if (args.split != args.nsplits - 1) else nprompts
        prompts = prompts[left:right]
        print(f"Creating prompts from {left} to {right}")
    
    # (re)start experiment
    os.makedirs(args.output_dir, exist_ok=True)
    start_point = 0 # if resuming, start from the last line of the file
    if os.path.exists(os.path.join(args.output_dir, f"results.jsonl")):
        with open(os.path.join(args.output_dir, f"results.jsonl"), "r") as f:
            for _ in f:
                start_point += 1
    print(f"Starting from {start_point}")

    # generate
    all_times = []
    with open(os.path.join(args.output_dir, f"results.jsonl"), "a") as f:
        for ii in range(start_point, len(prompts), args.batch_size):
            # generate chunk
            time0 = time.time()
            chunk_size = min(args.batch_size, len(prompts) - ii)
            results = generator.generate(
                prompts[ii:ii+chunk_size], 
                max_gen_len=args.max_gen_len, 
                temperature=args.temperature, 
                top_p=args.top_p
            )
            time1 = time.time()
            # time chunk
            speed = chunk_size / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta)) 
            all_times.append(time1 - time0)
            print(f"Generated {ii:5d} - {ii+chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            # log
            for prompt, result in zip(prompts[ii:ii+chunk_size], results):
                f.write(json.dumps({
                    "prompt": prompt, 
                    "result": result[len(prompt):],
                    "payload": generator.payload,
                    "speed": speed,
                    "eta": eta}) + "\n")
                f.flush()

            # change payload if payload_mode is random
            if args.payload_mode == 'random':
                generator.set_payload(random.randint(0, args.payload_max - 1))
    print(f"Average generation time per prompt: {np.sum(all_times) / (len(prompts) - start_point) :.2f}")

    if args.method_detect == 'same':
        args.method_detect = args.method
    if (not args.do_eval) or (args.method_detect not in ["openai", "maryland", "marylandz", "openaiz", "marylandE", "rs", "rsbh", "bch"]):
        raise ValueError('Unknown detect method!')
    
    # build watermark detector
    if args.method_detect == "openai":
        detector = OpenaiDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method_detect == "openaiz":
        detector = OpenaiDetectorZ(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key)
    elif args.method_detect == "maryland":
        detector = MarylandDetector(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)
    elif args.method_detect == "marylandz":
        detector = MarylandDetectorZ(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta)
    elif args.method_detect == "marylandE":
        detector = MarylandDetectorE(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta, payload_max=args.payload_max)
    elif args.method_detect == "rs":
        detector = RSDecoder(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta,
                             segments_num=args.segments_num, gf_segments_num=args.gf_segments_num, segment_bit=args.segment_bit)
    elif args.method_detect == "rsbh":
        detector = RSBHDecoder(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta,
                             segments_num=args.segments_num, gf_segments_num=args.gf_segments_num, segment_bit=args.segment_bit,
                             bh_mapping=bh_mapping)
    elif args.method_detect == "bch":
        detector = BCHDecoder(tokenizer, args.ngram, args.seed, args.seeding, args.hash_key, gamma=args.gamma, delta=args.delta, payload_max=args.payload_max)

    # load real completion
    if 'OpenGen' in args.prompt_path:
        results_orig = load_results(json_path=args.prompt_path, nsamples=args.nsamples, result_key="targets")
    elif 'C4' in args.prompt_path:
        results_orig = load_results(json_path=args.prompt_path, nsamples=args.nsamples, result_key='gold_completion')
    elif 'essays' in args.prompt_path:
        results_orig = load_results(json_path=args.prompt_path, nsamples=args.nsamples, result_key='gold_completion')
    else:
        results_orig = load_results(json_path=args.prompt_path, nsamples=args.nsamples, result_key="output")
    if args.split is not None:
        results_orig = results_orig[left:right]

    # evaluate
    results, corr_payloads = load_res_payload(json_path=os.path.join(args.output_dir, f"results.jsonl"), nsamples=args.nsamples, result_key="result")
    log_stats = []
    text_index = left if args.split is not None else 0
    all_times = []
    decoded_payloads = []
    with open(os.path.join(args.output_dir, 'scores.jsonl'), 'w') as f:
        for text, text_orig in tqdm.tqdm(zip(results, results_orig)):
            time0 = time.time()
            # compute watermark score
            if args.method_detect in ["rs", "rsbh", "bch"]:
                scores, num_tokens, segment_allocation_list = detector.get_aggregate_scores([text], scoring_method=args.scoring_method)
                payloads = detector.get_decoded_payload(scores)
                zscores, pvalues = get_pvalue_segment_based(scores, num_tokens)
                # make scores serializable and compatible with other situations
                scores = [[arr.tolist() for arr in scores[0]]]
            else:
                scores, num_tokens = detector.get_aggregate_scores([text], scoring_method=args.scoring_method, payload_max=args.payload_max)
                pvalues = detector.get_pvalues_from_aggr_scores(scores, num_tokens)
                # for compatibility when saving results
                segment_allocation_list = []

            # discard results with num_tokens < max_gen_len - 10
            # if num_tokens[0] < args.max_gen_len - 10:
            #     text_index += 1
            #     continue

            if args.method_detect not in ["rs", "rsbh", 'bch']:
                all_scores = [scores[0].tolist()]
            else:
                all_scores = [scores[0]]

            # save decoded payload, and correct pvalues if necessary
            if args.payload_max:
                if args.method_detect in ["rs", "rsbh", "bch"]:
                    decoded_payloads.extend(payloads)
                else:
                    payloads = np.argmin(pvalues, axis=1).tolist() # decoded payloads, shape: (bsz)
                    decoded_payloads.extend(payloads) # record decoded payloads
                    pvalues = pvalues[:,payloads][0].tolist() 
                    scores = [float(s[payload]) for s,payload in zip(scores,payloads)]
                    M = args.payload_max  
                    pvalues = [(1 - (1 - pvalue)**M) if pvalue > min(1 / M, 1e-5) else M * pvalue for pvalue in pvalues]
            else:
                payloads = [ 0 ] * len(pvalues)
                pvalues = pvalues[:,0].tolist()
                scores = [float(s[0]) for s in scores]

            time1 = time.time()
            all_times.append(time1 - time0)

            # log stats and write to file
            log_stat = {
                'text_index': text_index,
                'num_token': int(num_tokens[0]),
                'score': scores[0],
                # 'pvalue': pvalues[0], 
                'all_score': all_scores[0],
                'payload': payloads[0],
                'tokens_segment': segment_allocation_list,
            }
            # report zscore for segment based method, whereas report pvalue for traditional method like maryland and openai
            if args.method_detect in ['rs', 'rsbh', 'bch']:
                log_stat['zscore'] = zscores[0]
            else:
                log_stat['pvalue'] = pvalues[0]

            f.write(json.dumps(log_stat)+'\n') 

            # short log stat containing only one single for each field, to enable in running analysis 
            short_log_stat = {k: log_stat[k] for k in ['text_index', 'num_token', 'score', 'payload']}
            # report zscore for segment based method, whereas report pvalue for traditional method like maryland and openai
            if args.method_detect in ['rs', 'rsbh', 'bch']:
                short_log_stat['zscore'] = zscores[0]
            else:
                short_log_stat['pvalue'] = pvalues[0]
            log_stats.append(short_log_stat) # only append short_log_stat to log_stats to avoid memory issues
            text_index += 1

        df = pd.DataFrame(log_stats)

        with open(args.output_log_path, 'a') as logfile:
            # calculate bit acc
            bit_width = int(log2(args.payload_max))
            print(f"======================================================", file=logfile)
            if 'rs' in args.method:
                print(f"Method: {args.method}, Model: {args.model_name}, Dataset: {args.prompt_path} Parameters: n={args.gf_segments_num}, k={args.segments_num}, m={args.segment_bit}", file=logfile)
            else:
                print(f"Method: {args.method}, Model: {args.model_name}, Dataset: {args.prompt_path} Parameters: bit_num={bit_width}", file=logfile)
            print(f"delta: {args.delta}, gamma: {args.gamma}, T: {args.max_gen_len}", file=logfile)
            print(f">>> Scores: \n{df.describe(percentiles=[])}", file=logfile)
            print(f"Saved scores to {os.path.join(args.output_dir, 'scores.jsonl')}", file=logfile)
            print(f"Average decoding time per prompt: {np.sum(all_times) / (np.array(decoded_payloads).size) :.5f}", file=logfile)
            print(f"Decoded payloads:{decoded_payloads}", file=logfile)
            print(f"Accuracy: {np.sum(np.array(decoded_payloads) == np.array(corr_payloads)) / (np.array(decoded_payloads).size) :.4f}", file=logfile)
            sum_bitacc = sum([bitacc(res, corr, bit_width) for res, corr in zip(decoded_payloads, corr_payloads)])
            print(f"Bit accuracy: {sum_bitacc / (np.array(decoded_payloads).size) / bit_width}", file=logfile)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
