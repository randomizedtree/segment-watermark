from typing import List
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from math import log2
from galois import BCH
from generalizedReedSolomon.generalizedreedsolo import Generalized_Reed_Solomon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmGenerator():
    def __init__(self, 
            model: LlamaForCausalLM, 
            tokenizer: LlamaTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            seeding: str = 'hash',
            salt_key: int = 35317,
            payload: int = 0,
            model_name: str = 'guanaco'
        ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        # if max_sequence_length doesn't exist, use 2048 as default according to training process
        self.max_seq_len = model.config.max_sequence_length if hasattr(model.config, "max_sequence_length") else 2048
        # set pad id to 99999 if not exist
        self.pad_id = model.config.pad_token_id if model.config.pad_token_id else 99999
        self.eos_id = model.config.eos_token_id
        # watermark config
        self.ngram = ngram
        self.salt_key = salt_key
        self.seed = seed
        self.hashtable = torch.randperm(1000003)
        self.seeding = seeding 
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.payload = payload
        self.model_name = model_name
        if self.model_name == 'falcon':
            self.newline_idx = 193
        elif self.model_name == 'gemma':
            self.newline_idx = 108
        else:
            self.newline_idx = 13

        # whether the previous generated token is \n
        self.is_slash_n = False

    def hashint(self, integer_tensor: torch.LongTensor) -> torch.LongTensor:
        """Adapted from https://github.com/jwkirchenbauer/lm-watermarking"""
        return self.hashtable[integer_tensor.cpu() % len(self.hashtable)] # basically attributes each element a random int, based on its original value
    
    def get_seed_rng(
        self, 
        input_ids: torch.LongTensor # input_ids is likely to be ngrams in tensor like [x^-h, x^{-h+1},..., x^{-1}]
    ) -> int:
        """
        Seed RNG with hash of input_ids.
        Adapted from https://github.com/jwkirchenbauer/lm-watermarking
        """
        if self.seeding == 'hash':
            seed = self.seed
            for i in input_ids:
                seed = (seed * self.salt_key + i.item()) % (2 ** 64 - 1)
        elif self.seeding == 'additive':
            seed = self.salt_key * torch.sum(input_ids).item()
            seed = self.hashint(seed)
        elif self.seeding == 'skip':
            seed = self.salt_key * input_ids[0].item()
            seed = self.hashint(seed)
        elif self.seeding == 'min':
            seed = self.hashint(self.salt_key * input_ids)
            seed = torch.min(seed).item()
        return seed

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """
        
        # num of prompts each batch
        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(device).long() # init output with padding tokens
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long() # fill `tokens` with prompt tokens
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            ) # (bsz, tokens_len, vocab_size)
            ngram_tokens = tokens[:, cur_pos-self.ngram:cur_pos]
            next_toks = self.sample_next(outputs.logits[:, -1, :], ngram_tokens, temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks) # only updates those with cur_pos is a padding token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded
    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding. i.e. for each batch, the ngrams for current pos 
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """ Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p # for pos p, mask[p] = True when sum(probs_sort[:p]) > top_p, i.e. sum of previous 0~(p-1) probs exceed top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # regularization after mask out minor probs 
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

    def set_payload(self, new_payload):
        """Update payload, used when payload is randomized in exp"""
        self.payload = new_payload

class OpenaiGenerator(WmGenerator):
    """ Generate text using LLaMA and Aaronson's watermarking method. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to generate V random number r between [0,1]
        - select argmax ( r^(1/p) )
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            for ii in range(ngram_tokens.shape[0]): # batch of texts
                # seed with hash of ngram tokens
                seed = self.get_seed_rng(ngram_tokens[ii])
                self.rng.manual_seed(seed)
                # generate rs randomly between [0,1]
                rs = torch.rand(self.tokenizer.vocab_size, generator=self.rng) # n
                rs = rs.roll(-self.payload)
                rs = torch.Tensor(rs).to(probs_sort.device)
                rs = rs[probs_idx[ii]]  # reorder rs so that the r-value which corresponds to the vocab with largest prob will appear first
                # compute r^(1/p)
                probs_sort[ii] = torch.pow(rs, 1/probs_sort[ii])
            # select argmax ( r^(1/p) )
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

class MarylandGenerator(WmGenerator):
    """ Generate text using LLaMA and Maryland's watemrarking method. """
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            delta: float = 1.0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma
        self.delta = delta

    def sample_next(
        self,
        logits: torch.FloatTensor, # (bsz, vocab_size): logits for last token
        ngram_tokens: torch.LongTensor, # (bsz, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ) -> torch.LongTensor:
        """
        From ngram tokens, select the next token based on the following:
        - hash the ngram tokens and get a seed
        - use the seed to partition the vocabulary into greenlist (gamma*V words) and blacklist 
        - add delta to greenlist words' logits
        payload (the message) is encoded by shifting the secret vector r by `payload`.
        """
        logits = self.logits_processor(logits, ngram_tokens)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        if next_token == self.newline_idx:
            self.is_slash_n = True
        else:
            self.is_slash_n = False
        return next_token

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to mask out words in greenlist."""
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            vocab_permutation = torch.randperm(vocab_size, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * vocab_size)] # gamma * n
            bias = torch.zeros(vocab_size).to(logits.device) # n
            bias[greenlist] = self.delta
            bias = bias.roll(-self.payload)  
            logits[ii] += bias # add bias to greenlist words
        return logits


# ===============================================================================================================
# =============================     Kirchenbauer et al. Extended cyclic shift   =================================
# ===============================================================================================================

class MarylandGeneratorE(MarylandGenerator):
    def __init__(
            self,
            *args,
            payload_max,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.payload_max = payload_max


    def logits_processor(self, logits, ngram_tokens):
        """
        Process logits to mask out words in greenlist.
        NOTE: currently we 
              - use gamma based on the extended random bias, 
              - cyclic shift it based on message
              - truncate it to vocab_size
              So it is possible that in final bias, the greenlist num is not (gamma*vocab_size)
        """
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            # length of the entire r to roll
            r_length = max(self.payload_max, vocab_size)
            vocab_permutation = torch.randperm(r_length, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * payload_max, index of greenlist token
            bias = torch.zeros(r_length).to(logits.device) # payload_max
            bias[greenlist] = self.delta
            bias = bias.roll(-self.payload)  
            logits[ii] += bias[:vocab_size] # truncate the bias tensor to vocab_size and add to logits

            if self.is_slash_n:
                logits[ii][self.newline_idx] -= 100
            logits[ii][self.eos_id] = -65000
        return logits


# ===============================================================================================================
# ======================================     BCH with segments            =======================================
# ===============================================================================================================


class BCHGenerator(MarylandGenerator):
    def __init__(
            self,
            *args,
            payload_max,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        # BCH table
        self.payload_max = payload_max
        # bit num
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
        # 1 will become 1, 0, ..., 0, MSB is at the end
        self.payload_bits = [ ((self.payload >> i) & 1) for i in range(self.bitwidth)]
        # pad with 0
        if len(self.payload_bits) < self.k:
            self.payload_bits.extend([0 for i in range(self.k - len(self.payload_bits))]) 

        self.BCH = BCH(self.n, self.k)
        self.encoded_payload_bits = [int(i) for i in self.BCH.encode(self.payload_bits)]

        # pad another bit to make it divisible by 4 or 8
        self.encoded_payload_bits.append(0)

        # how many segments do we have
        self.segments_num = int(len(self.encoded_payload_bits) / self.segment_bit)

        self.segments = []
        for i in range(self.segments_num):
            bits_for_this_seg = self.encoded_payload_bits[(i*self.segment_bit) : ((i+1)*self.segment_bit)]
            self.segments.append(int(''.join(map(str, bits_for_this_seg)), 2))

    def logits_processor(self, logits, ngram_tokens):
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            # generate r
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            # r_length will be vocab_size unless 2**self.segment_bits > vocab_size, usually satisfied under our exp
            r_length = vocab_size
            vocab_permutation = torch.randperm(r_length, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * payload_max, index of greenlist token
            bias = torch.zeros(r_length).to(logits.device) # payload_max
            bias[greenlist] = self.delta

            # assign current token to one segment
            random_int = torch.randint(low=0, high=self.segments_num, size=(1,), generator=self.rng).item() 
            bias = bias.roll(-self.segments[random_int])

            bias = bias[:vocab_size]  # truncate the bias tensor to vocab_size and add to logits

            logits[ii] += bias[:vocab_size] 

            if self.is_slash_n:
                logits[ii][self.newline_idx] -= 100
            logits[ii][self.eos_id] = -65000
        return logits

    def set_payload(self, new_payload):
        """ update payload used to embed """
        self.payload = new_payload
        # 1 will become 1, 0, ..., 0, MSB is at the end
        self.payload_bits = [ ((self.payload >> i) & 1) for i in range(self.bitwidth)]
        # pad with 0
        if len(self.payload_bits) < self.k:
            self.payload_bits.extend([0 for i in range(self.k - len(self.payload_bits))]) 

        self.encoded_payload_bits = [int(i) for i in self.BCH.encode(self.payload_bits)]

        # pad another bit to make it divisible by 4 or 8
        self.encoded_payload_bits.append(0)

        self.segments = []
        for i in range(self.segments_num):
            bits_for_this_seg = self.encoded_payload_bits[(i*self.segment_bit) : ((i+1)*self.segment_bit)]
            self.segments.append(int(''.join(map(str, bits_for_this_seg)), 2))

# ===============================================================================================================
# ==========     Combining ReedSolomon ECC with rolling based method                 ============================
# ===============================================================================================================


class RSGenerator(MarylandGenerator):
    def __init__(
            self,
            *args,
            segments_num,
            gf_segments_num,
            segment_bit,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        # total number of segments, or k in RS code
        self.segments_num = segments_num
        # total number of segments after RS, or n in RS code
        self.gf_segments_num = gf_segments_num
        # the # of bits within one segment, m in RS code (GF(q^m))
        self.segment_bit = segment_bit

        # bidwidth = segment_bit * segments_num
        self.bitwidth = self.segments_num * self.segment_bit

        # 1. divide original message into segments
        mask = 2 ** self.segment_bit - 1
        self.segments = [
            (self.payload >> (self.segment_bit * i)) & mask for i in range(self.segments_num)
        ]

        # 2. encode segments with RS
        self.rs = Generalized_Reed_Solomon(
            field_size=2,
            message_length=self.gf_segments_num,
            payload_length=self.segments_num,
            symbol_size=self.segment_bit,
            p_factor=1
        )

        self.gf_segments = [int(i) for i in self.rs.encode(self.segments)]


    def logits_processor(self, logits, ngram_tokens):
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            # generate r
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            r_length = vocab_size
            vocab_permutation = torch.randperm(r_length, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * payload_max, index of greenlist token
            bias = torch.zeros(r_length).to(logits.device) # payload_max
            bias[greenlist] = self.delta

            # assign current token to one of the segment
            random_int = torch.randint(low=0, high=self.gf_segments_num, size=(1,), generator=self.rng).item() 
            bias = bias.roll(-self.gf_segments[random_int])

            bias = bias[:vocab_size]  # truncate the bias tensor to vocab_size and add to logits

            logits[ii] += bias[:vocab_size] 

            if self.is_slash_n:
                logits[ii][self.newline_idx] -= 100
            logits[ii][self.eos_id] = -65000
        return logits

    def set_payload(self, new_payload):
        self.payload = new_payload
        
        # 1. divide original message into segments
        mask = 2 ** self.segment_bit - 1
        self.segments = [
            (self.payload >> (self.segment_bit * i)) & mask for i in range(self.segments_num)
        ]
        # 2. encode segments with RS
        self.gf_segments = [int(i) for i in self.rs.encode(self.segments)]


# ===============================================================================================================
# ==========     ReedSolomon ECC with balanced hash                                  ============================
# ===============================================================================================================

class RSBHGenerator(MarylandGenerator):
    def __init__(
            self,
            *args,
            segments_num,
            gf_segments_num,
            segment_bit,
            bh_mapping,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        # FIXME: currently only support fixed RS scheme
        # total number of segments, or k in RS code
        self.segments_num = segments_num
        # total number of segments after RS, or n in RS code
        self.gf_segments_num = gf_segments_num
        # the # of bits within one segment, m in RS code (GF(q^m))
        self.segment_bit = segment_bit

        # bidwidth = segment_bit * segments_num
        self.bitwidth = self.segments_num * self.segment_bit

        # balanced hash mapping, we generate this in an offline manner
        self.mapping = bh_mapping

        # 1. divide original message into segments
        mask = 2 ** self.segment_bit - 1
        self.segments = [
            (self.payload >> (self.segment_bit * i)) & mask for i in range(self.segments_num)
        ]

        # 2. encode segments with RS
        self.rs = Generalized_Reed_Solomon(
            field_size=2,
            message_length=self.gf_segments_num,
            payload_length=self.segments_num,
            symbol_size=self.segment_bit,
            p_factor=1
        )

        self.gf_segments = [int(i) for i in self.rs.encode(self.segments)]


    def logits_processor(self, logits, ngram_tokens):
        bsz, vocab_size = logits.shape
        logits = logits.clone()
        for ii in range(ngram_tokens.shape[0]): # batch of texts
            # generate r
            seed = self.get_seed_rng(ngram_tokens[ii])
            self.rng.manual_seed(seed)
            r_length = vocab_size
            vocab_permutation = torch.randperm(r_length, generator=self.rng)
            greenlist = vocab_permutation[:int(self.gamma * r_length)] # gamma * payload_max, index of greenlist token
            bias = torch.zeros(r_length).to(logits.device) # payload_max
            bias[greenlist] = self.delta

            # assign current token to one segment based on mapping
            random_int = self.mapping[ngram_tokens[ii][0].cpu().item()]
            bias = bias.roll(-self.gf_segments[random_int])

            bias = bias[:vocab_size]  # truncate the bias tensor to vocab_size and add to logits

            logits[ii] += bias[:vocab_size] 

            if self.is_slash_n:
                logits[ii][self.newline_idx] -= 100
            logits[ii][self.eos_id] = -65000
        return logits

    def set_payload(self, new_payload):
        self.payload = new_payload
        
        # 1. divide original message into segments
        mask = 2 ** self.segment_bit - 1
        self.segments = [
            (self.payload >> (self.segment_bit * i)) & mask for i in range(self.segments_num)
        ]
        # 2. encode segments with RS
        self.gf_segments = [int(i) for i in self.rs.encode(self.segments)]