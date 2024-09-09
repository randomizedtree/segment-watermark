from typing import List
import numpy as np
from numpy._core.multiarray import array as array
from scipy import special
import torch
from transformers import LlamaTokenizer
import json
from math import log2, sqrt
from galois import BCH
from generalizedReedSolomon.generalizedreedsolo import Generalized_Reed_Solomon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmDetector():
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317
        ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] 
    
    def get_seed_rng(self, input_ids: List[int]) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids)
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0]
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed)
        return seed

    def aggregate_scores(self, scores: List[List[np.array]], aggregation: str = 'mean') -> List[float]:
        """Aggregate scores along a text."""
        scores = np.asarray(scores) # scores: (bsz, text_tokens, num_of_candidate_payloads)
        if aggregation == 'sum':
           return [ss.sum(axis=0) for ss in scores]
        elif aggregation == 'mean':
            return [ss.mean(axis=0) if ss.shape[0]!=0 else np.ones(shape=(self.vocab_size)) for ss in scores]
        elif aggregation == 'max':
            return [ss.max(axis=0) for ss in scores]
        else:
             raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_scores_by_t(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
        payload_max: int = 0
    ) -> List[np.array]:
        """
        Get score increment for each token in list of texts.
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score increments for every token and payload] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            rts = []
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1': # v1: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}]
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2': # v2: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}, x^0]
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                # NOTE: modify the below to truncate the rt to 0~(payload_max-1) as I expect all payloads < payload_max
                rt = rt.numpy()[:payload_max] # rt: contribution of token t on each payloads
                rts.append(rt)
            score_lists.append(rts)
        return score_lists # scores_lists: (bsz, text_tokens, num_of_candidate_payloads)

    def get_aggregate_scores(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
        payload_max: int = 0
    ):
        """
        Get score for each payload in list of texts (aggregated across all tokens generated using sum)
        Args:
            texts: list of texts
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
            payload_max: maximum number of messages 
        Output:
            score_lists: list of [np array of score for each payload] for each text
            ntoks_arr: list of [# of generated tokens] for each text
        """
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]
        score_lists = []
        ntoks_arr = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            rt_aggr = torch.zeros(payload_max) # init aggregate scores as all 0
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1': # v1: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}]
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2': # v2: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}, x^0]
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt[:payload_max] # rt: contribution of token t on each payloads
                rt_aggr += rt # add contribution of token t to rt_aggr
            score_lists.append(rt_aggr.numpy())
            ntoks_arr.append(total_len - start_pos) 
        return score_lists, np.asarray(ntoks_arr) 

    def get_pvalues(
            self, 
            scores: List[np.array], 
            eps: float=1e-200
        ) -> np.array:
        """
        Get p-value for each text.
        Args:
            score_lists: list of [list of score increments for each token] for each text
        Output:
            pvalues: np array of p-values for each text and payload
        """
        pvalues = []
        scores = np.asarray(scores) # bsz x ntoks x payload_max
        for ss in scores:
            ntoks = ss.shape[0]
            scores_by_payload = ss.sum(axis=0) if ntoks!=0 else np.zeros(shape=ss.shape[-1]) # payload_max
            pvalues_by_payload = [self.get_pvalue(score, ntoks, eps=eps) for score in scores_by_payload]
            pvalues.append(pvalues_by_payload)
        return np.asarray(pvalues) # bsz x payload_max

    def get_pvalues_from_aggr_scores(
        self,
        scores: np.array,
        ntoks_arr: np.array,
        eps: float=1e-200
    ) -> np.array:
        """
        Get p-value for each text.
        Args:
            scores: list of [list of scores (aggr by sum) for each payload] for each text
            ntoks_arr: list of num of tokens generated for each text
        Output:
            pvalues: np array of p-values for each text and payload
        """
        pvalues = []
        scores = np.asarray(scores) # bsz x payload_max
        ntoks_arr = np.asarray(ntoks_arr) # bsz
        for ss, ntoks in zip(scores, ntoks_arr):
            pvalues_by_payload = [self.get_pvalue(score, ntoks, eps=eps) for score in ss]
            pvalues.append(pvalues_by_payload)
        return np.asarray(pvalues) # bsz x payload_max


    def get_pvalues_by_t(self, scores: List[float]) -> List[float]: 
        """Get p-value for each text."""
        pvalues = []
        cum_score = 0
        cum_toks = 0
        for ss in scores:
            cum_score += ss
            cum_toks += 1
            pvalue = self.get_pvalue(cum_score, cum_toks)
            pvalues.append(pvalue)
        return pvalues

    def save_rt(self, path: str, rt: torch.tensor, cur_pos: int) -> None:
        with open(path, 'a') as f:
            f.write(json.dumps({"pos": cur_pos, "data": rt.tolist()}) + "\n")
    
    def score_tok(self, ngram_tokens: List[int], token_id: int):
        """ for each token in the text, compute the score increment """
        raise NotImplementedError
    
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ compute the p-value for a couple of score and number of tokens """
        raise NotImplementedError


class MarylandDetector(WmDetector):
    """ Modified detector for Kirchenbauer et al. using Multinomial hypothesis testing """
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0 
        The last line shifts the scores by token_id. 
        ex: scores[0] = 1 if token_id in greenlist else 0
            scores[1] = 1 if token_id in (greenlist shifted of 1) else 0
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id) 
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)

class MarylandDetectorZ(WmDetector):
    """ original Kirchenbauer et al. detector"""
    def __init__(self, 
            tokenizer: LlamaTokenizer,
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ same as MarylandDetector but using zscore """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
        scores[greenlist] = 1
        return scores.roll(-token_id)
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a normal distribution """
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    
class OpenaiDetector(WmDetector):
    """ Modified detector of  Aaronson et al. using no approximation """
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = -log(1 - rt[token_id]])
        The last line shifts the scores by token_id. 
        ex: scores[0] = r_t[token_id]
            scores[1] = (r_t shifted of 1)[token_id]
            ...
        The score for each payload will be given by scores[payload]
        """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log().roll(-token_id)
        return scores
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)

class OpenaiDetectorZ(WmDetector):
    """ Original detector of Aaronson et al. using z scores """
    def __init__(self, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, seeding, salt_key, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ same as OpenaiDetector but using zscore """
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log().roll(-token_id)
        return scores
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a normal distribution """
        mu0 = 1
        sigma0 = np.pi / np.sqrt(6)
        zscore = (score/ntoks - mu0) / (sigma0 / np.sqrt(ntoks))
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    

# ===============================================================================================================
# =============================     Kirchenbauer et al. Extended cyclic shift   =================================
# ===============================================================================================================

class MarylandDetectorE(MarylandDetector):
    def __init__(self, *args, payload_max, **kwargs):
        super().__init__(*args, **kwargs)
        self.payload_max = payload_max

    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        r_length = max(self.payload_max, self.vocab_size)
        scores = torch.zeros(r_length) # scores tensor of all possible payloads
        vocab_permutation = torch.randperm(r_length, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id) 


# ===============================================================================================================
# ======================================     BCH with segments            =======================================
# ===============================================================================================================


class BCHDecoder(MarylandDetector):
    def __init__(
            self, 
            *args,
            payload_max,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.payload_max = payload_max
        self.bitwidth = int(log2(self.payload_max)) # bit width of payload

        if self.bitwidth not in [12, 16, 20, 24, 28, 32]:
            raise ValueError("BCH generator: Not supported bit width")
        
        self.scheme_map = {
            12: (31, 16, 4),
            16: (31, 16, 4),
            20: (31, 21, 4),
            24: (31, 26, 4),
            28: (63, 30, 8),
            32: (63, 36, 8)
        }

        self.n = self.scheme_map[self.bitwidth][0]
        self.k = self.scheme_map[self.bitwidth][1]
        # how many bits in one segment
        self.segment_bit = self.scheme_map[self.bitwidth][2]
        self.segments_num = int((self.n + 1) / self.segment_bit)

        self.BCH = BCH(self.n, self.k)
    
    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        r_length = self.vocab_size
        scores = torch.zeros(r_length) # scores tensor of all possible payloads
        vocab_permutation = torch.randperm(r_length, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id) 

    def get_aggregate_scores(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
    ):
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]

        score_lists = [[] for i in range(bsz)]
        ntoks_arr = []
        
        random_int_list = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            # aggr scores for each payload segments
            rt_aggr_list = [torch.zeros(2 ** self.segment_bit) for i in range(self.segments_num)]
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1': # v1: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}]
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2': # v2: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}, x^0]
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt[:(2**self.segment_bit)] # rt: contribution of token t on each payloads

                # assign current token to m1 or m2
                random_int = torch.randint(low=0, high=self.segments_num, size=(1,), generator=self.rng).item() 
                rt_aggr_list[random_int] += rt

                random_int_list.append(random_int)

            for i in range(self.segments_num):
                score_lists[ii].append(rt_aggr_list[i].numpy())
            # num of scores vecctor being sumed is (total_len - 1 - start_pos + 1 = total_len - start_pos)
            ntoks_arr.append(total_len - start_pos) 
        return score_lists, np.asarray(ntoks_arr), random_int_list

    def get_decoded_payload(self, scores: List[List[np.ndarray]]) -> List[int]:
        """
        get decoded payload from scores (bsz, gf_segment_nums, num_of_candidate_payloads_in_each_segment)
        :return: List[int], list of decoded payload of each sample
        """
        bsz = len(scores) 
        payloads = []
        # for each sample
        for b_idx in range(bsz):
            max_value_of_segments = []
            # for each segment inside one sample
            for i in range(self.segments_num):
                max_value_of_segments.append(np.argmax(scores[b_idx][i], axis=-1).item())

            payload = 0

            # convert [14, 15, 17, ..., 19] like vector to [1, 1, 0, 0, ..., 1] like vector that can be used by BCH
            binary_v = []
            for i in max_value_of_segments:
                binary_v.extend(format(i, 'b').zfill(self.segment_bit))
            # get rid of the last padding bit (second pad to make 31/63 to 32/64)
            binary_v = binary_v[:-1]
            try:
                payload_in_bits = [int(i) for i in self.BCH.decode(binary_v)]
                # get rid of first padding bits
                payload_in_bits[:self.bitwidth]
            except (ZeroDivisionError, IndexError) as e:
                # directly return non-corrected payload when error
                print(f"{e} in BCH decode!")
                payload_in_bits = binary_v[:self.bitwidth]

            for i in range(self.bitwidth):
                payload += int(payload_in_bits[i]) << i

            payloads.append(payload)

        return payloads

# ===============================================================================================================
# ==========     Combining ReedSolomon ECC with rolling based method           ==================================
# ===============================================================================================================

class RSDecoder(MarylandDetector):
    def __init__(
            self, 
            *args,
            segments_num,
            gf_segments_num,
            segment_bit,
            **kwargs):
        super().__init__(*args, **kwargs)
        # total number of segments, or k in RS code
        self.segments_num = segments_num
        # total number of segments after RS, or n in RS code
        self.gf_segments_num = gf_segments_num
        # the # of bits within one segment, or m in RS code (GF(q^m))
        self.segment_bit = segment_bit
        # bitwidth
        self.bitwidth = self.segments_num * self.segment_bit

        self.rs = Generalized_Reed_Solomon(
            field_size=2,
            message_length=self.gf_segments_num,   # n
            payload_length=self.segments_num,      # k
            symbol_size=self.segment_bit,          # m
            p_factor=1
        )

    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        r_length = self.vocab_size
        scores = torch.zeros(r_length) # scores tensor of all possible payloads
        vocab_permutation = torch.randperm(r_length, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id) 

    def get_aggregate_scores(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
    ):
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]

        score_lists = [[] for i in range(bsz)]
        ntoks_arr = []
        random_int_list = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            # aggr scores for each payload segments
            rt_aggr_list = [torch.zeros(2 ** self.segment_bit) for i in range(self.gf_segments_num)]
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1': # v1: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}]
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2': # v2: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}, x^0]
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt[:(2**self.segment_bit)] # rt: contribution of token t on each payloads
                # determine which segment current token is embedding
                random_int = torch.randint(low=0, high=self.gf_segments_num, size=(1,), generator=self.rng).item() 
                rt_aggr_list[random_int] += rt

                random_int_list.append(random_int)

            for i in range(self.gf_segments_num):
                score_lists[ii].append(rt_aggr_list[i].numpy())
            # num of scores vecctor being sumed is (total_len - 1 - start_pos + 1 = total_len - start_pos)
            ntoks_arr.append(total_len - start_pos) 
        return score_lists, np.asarray(ntoks_arr), random_int_list

    def get_decoded_payload(self, scores: List[List[np.ndarray]]) -> List[int]:
        """
        get decoded payload from scores (bsz, gf_segment_nums, num_of_candidate_payloads_in_each_segment)
        :return: List[int], list of decoded payload of each sample
        """
        bsz = len(scores) 
        payloads = []
        # for each sample
        for b_idx in range(bsz):
            max_value_of_segments = []
            # for each segment inside one sample
            for i in range(self.gf_segments_num):
                max_value_of_segments.append(np.argmax(scores[b_idx][i], axis=-1).item())

            payload = 0
            try:
                payload_in_segs = self.rs.decode(max_value_of_segments)
            except (ZeroDivisionError, IndexError) as e:
                # directly return non-corrected payload when error
                print(f"{e} in RS decode!")
                payload_in_segs = max_value_of_segments[:self.segments_num]

            for i in range(self.segments_num):
                payload += int(payload_in_segs[i]) << (i * self.segment_bit)

            payloads.append(payload)

        return payloads


# ===============================================================================================================
# ==========     ReedSolomon ECC with balanced hash                            ==================================
# ===============================================================================================================

class RSBHDecoder(MarylandDetector):
    def __init__(
            self, 
            *args,
            segments_num,
            gf_segments_num,
            segment_bit,
            bh_mapping,
            **kwargs):
        super().__init__(*args, **kwargs)
        # total number of segments, or k in RS code
        self.segments_num = segments_num
        # total number of segments after RS, or n in RS code
        self.gf_segments_num = gf_segments_num
        # the # of bits within one segment, or m in RS code (GF(q^m))
        self.segment_bit = segment_bit
        # bitwidth
        self.bitwidth = self.segments_num * self.segment_bit

        # balance hash mapping
        self.mapping = bh_mapping

        self.rs = Generalized_Reed_Solomon(
            field_size=2,
            message_length=self.gf_segments_num,   # n
            payload_length=self.segments_num,      # k
            symbol_size=self.segment_bit,          # m
            p_factor=1
        )



    def score_tok(self, ngram_tokens, token_id):
        seed = self.get_seed_rng(ngram_tokens)
        self.rng.manual_seed(seed)
        r_length = self.vocab_size
        scores = torch.zeros(r_length) # scores tensor of all possible payloads
        vocab_permutation = torch.randperm(r_length, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores.roll(-token_id) 

    def get_aggregate_scores(
        self, 
        texts: List[str], 
        scoring_method: str="none",
        ntoks_max: int = None,
    ):
        bsz = len(texts)
        tokens_id = [self.tokenizer.encode(x, add_special_tokens=False) for x in texts]
        if ntoks_max is not None:
            tokens_id = [x[:ntoks_max] for x in tokens_id]

        score_lists = [[] for i in range(bsz)]
        ntoks_arr = []
        # FIXME: only support batch size 1
        random_int_list = []
        for ii in range(bsz):
            total_len = len(tokens_id[ii])
            start_pos = self.ngram +1
            # aggr scores for each payload segments
            rt_aggr_list = [torch.zeros(2 ** self.segment_bit) for i in range(self.gf_segments_num)]
            seen_ntuples = set()
            for cur_pos in range(start_pos, total_len):
                ngram_tokens = tokens_id[ii][cur_pos-self.ngram:cur_pos] # h
                if scoring_method == 'v1': # v1: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}]
                    tup_for_unique = tuple(ngram_tokens)
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                elif scoring_method == 'v2': # v2: only count unique [x^{-h}, x^{-h+1}, ..., x^{-1}, x^0]
                    tup_for_unique = tuple(ngram_tokens + tokens_id[ii][cur_pos:cur_pos+1])
                    if tup_for_unique in seen_ntuples:
                        continue
                    seen_ntuples.add(tup_for_unique)
                rt = self.score_tok(ngram_tokens, tokens_id[ii][cur_pos]) 
                rt = rt[:(2**self.segment_bit)] # rt: contribution of token t on each payloads

                # determine the segment current token embeds using mapping
                random_int = self.mapping[ngram_tokens[0]]
                rt_aggr_list[random_int] += rt

                random_int_list.append(random_int)

            for i in range(self.gf_segments_num):
                score_lists[ii].append(rt_aggr_list[i].numpy())
            ntoks_arr.append(total_len - start_pos) 
        return score_lists, np.asarray(ntoks_arr), random_int_list

    def get_decoded_payload(self, scores: List[List[np.ndarray]]) -> List[int]:
        """
        get decoded payload from scores (bsz, gf_segment_nums, num_of_candidate_payloads_in_each_segment)
        :return: List[int], list of decoded payload of each sample
        """
        bsz = len(scores) 
        payloads = []
        # for each sample
        for b_idx in range(bsz):
            max_value_of_segments = []
            # for each segment inside one sample
            for i in range(self.gf_segments_num):
                max_value_of_segments.append(np.argmax(scores[b_idx][i], axis=-1).item())

            payload = 0
            try:
                payload_in_segs = self.rs.decode(max_value_of_segments)
            except (ZeroDivisionError, IndexError) as e:
                print(f"{e} in RS decode!")
                payload_in_segs = max_value_of_segments[:self.segments_num]

            for i in range(self.segments_num):
                payload += int(payload_in_segs[i]) << (i * self.segment_bit)

            payloads.append(payload)

        return payloads

# ===============================================================================================================
# ==========     zscore & pvalue for all segment based method                  ==================================
# ===============================================================================================================

def get_pvalue_segment_based(scores: List[List[np.ndarray]], token_num: np.ndarray, eps: float=1e-200):
    """
    :param score: shape is (bsz, gf_segments_num, 2**segment_bit), i.e. COUNT array in paper for each sample in batch
    :return: zscore and pvalue for each sample in batch
    """
    bsz = len(scores)
    zscores = []
    pvalues = []
    for ii in range(bsz):
        max_of_sum = sum([np.max(score_of_seg) for score_of_seg in scores[ii]])
        zscore = (max_of_sum - 0.5*token_num[ii]) / (0.5 * sqrt(token_num[ii]))
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        zscores.append(zscore)
        pvalues.append(max(pvalue, eps))

    return zscores, pvalues
