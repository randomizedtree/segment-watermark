import json
from typing import List, Tuple

def bool_inst(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected in args')

def bitacc(res, accu, max_bit):
    res_bin = format(res, 'b').zfill(max_bit)
    accu_bin = format(accu, 'b').zfill(max_bit)
    acc_v = sum([res_bin[i] == accu_bin[i] for i in range(max_bit)])
    return acc_v

def load_prompts(
    json_path: str, 
    nsamples: int=None) -> List[str]:
    """
    choose first n samples from a dataset
    """
    all_prompts = []
    with open(json_path, "r") as f:
        p = f.readline()
        while p:
            all_prompts.append(json.loads(p)['prefix'])
            p = f.readline()
    prompts = all_prompts[:nsamples]
    print(f"Extracted first {len(prompts)} prompts from {len(all_prompts)}")
    return prompts

def load_prompts_by_index(
    json_path: str,
    samples_indices: List[int]) -> List[str]:
    """
    Load specific prompts with prompt index, used when debugging
    """
    all_prompts = []
    with open(json_path, "r") as f:
        p = f.readline()
        prompt_index = 0
        while p:
            if prompt_index in samples_indices:
                all_prompts.append(json.loads(p)['prefix'])
            p = f.readline()
            prompt_index += 1
    print(f"Extracted given {len(all_prompts)} prompts from {len(all_prompts)}")
    return all_prompts

def load_results(json_path: str, nsamples: int=None, result_key: str='result') -> List[str]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] # load jsonl
    new_prompts = [o[result_key] for o in prompts]
    new_prompts = new_prompts[:nsamples]
    return new_prompts

def load_res_payload(json_path: str, nsamples: int=None, result_key: str='result') -> Tuple[List[str], List[int]]:
    with open(json_path, "r") as f:
        if json_path.endswith('.json'):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()] # load jsonl
    # load generated text
    new_prompts = [o[result_key] for o in prompts]
    new_prompts = new_prompts[:nsamples]
    # load correct payload
    corr_payload = [o['payload'] for o in prompts]
    corr_payload = corr_payload[:nsamples]
    return new_prompts, corr_payload