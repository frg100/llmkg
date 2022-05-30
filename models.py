import os
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from abc import ABC, abstractmethod

import transformers
import scipy
from happytransformer import HappyTextToText, TTSettings

class LargeLanguageModel(ABC):
    def __init__(self, name=None, device='cpu', verbose = 0):
        self._name = name
        self._verbose = verbose

        self._device = device
        
        self._grammar_model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
        self._grammar_model_args = TTSettings(num_beams=5, min_length=1)
        
        self._built = False
        
    ####### PUBLIC #######
    @property
    def name(self):
        return self._name

    @property
    def device(self):
        return self._device
    
    def correct_grammar(self, sentence):
        return self._grammar_model.generate_text(f"grammar: {sentence}", args=self._grammar_model_args).text
    
    ####### PRIVATE #######
    
    @abstractmethod
    def batch_perplexity(self, eid):
        """
        A function that calculates a batch perplexity for a set of
        samples.
        """
        
    def test_discriminatory_power(self, graph, num_examples):
        pos_samples = graph.sample_english(num_examples)
        neg_samples = graph.sample_english(num_examples, negative=True)
        
        pos_perplexity = self.batch_perplexity(pos_samples)
        neg_perplexity = self.batch_perplexity(neg_samples)
        
        return scipy.stats.ttest_ind(pos_perplexity, neg_perplexity).pvalue

    def experiment(self, graph, num_runs, examples_per_run):
        save_path = f"results_{self.name}_{graph.name}_{num_runs}x{examples_per_run}.txt"
        
        p_vals = []
        for i in range(num_runs):
            p_val = self.test_discriminatory_power(graph, examples_per_run)
            print(f"The power of {self.name} on {graph.name} ({examples_per_run} samples) is: [{p_val}]")
            p_vals.append(str(p_val))

        with open(save_path, "w") as outfile:
            outfile.write("\n".join(p_vals))

        return p_vals
        
    @property
    def _is_built(self):
        return self._built


from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class GPT2(LargeLanguageModel):
    def __init__(self, model_id="gpt2", device='cpu', verbose = 0):
        super().__init__(name=f'GPT2_{model_id}', device=device, verbose = verbose)
        
        self._model_id = model_id
        self._model = GPT2LMHeadModel.from_pretrained(self._model_id).to(device)
        self._tokenizer = GPT2TokenizerFast.from_pretrained(self._model_id)
        
        self._verbose = verbose
        
        self._built = True
        
    def batch_perplexity(self, samples):
        if self._verbose >= 1:
            print(f"[{self._model_id}] Calculating perplexity for {len(samples)} samples")
        perplexities = []
        for sample in tqdm(samples):
            sample = self.correct_grammar(sample)
            encoding = self._tokenizer(sample, return_tensors="pt")
            num_tokens = encoding.input_ids.shape[1]

            nlls = []
            for end_loc in range(1, num_tokens):
                input_ids = encoding.input_ids[:, 0:end_loc].to(self.device)
                target_ids = input_ids.clone().to(self.device)

                with torch.no_grad():
                    outputs = self._model(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs[0] * end_loc

            nlls.append(neg_log_likelihood)

            perplexity = torch.exp(torch.stack(nlls).sum() / num_tokens)
            if self._verbose >= 2:
                print(f"[{self._model_id}] Sample <{sample}> has perplexity [{perplexity}]")
            perplexities.append(perplexity.item())

        if self._verbose >= 1:
            print(f"[{self._model_id}] Final average perplexity: {sum(perplexities)/len(perplexities)}")

        return perplexities 

from transformers import BertForMaskedLM, BertTokenizer
from torch.nn import functional as F


class BERT(LargeLanguageModel):
    def __init__(self, name='BERT', device='cpu', verbose = 0):
        super().__init__(name=name, device=device, verbose = verbose)
        
        self._name = name
        
        self._model_id = 'bert-base-cased'
        self._model = BertForMaskedLM.from_pretrained(self._model_id).to(device)
        self._tokenizer = BertTokenizer.from_pretrained(self._model_id)
        
        self._verbose = verbose
        
        self._built = True
        
    def batch_perplexity(self, samples):
        if self._verbose >= 1:
            print(f"[{self._name}] Calculating perplexity for {len(samples)} samples")
        perplexities = []
        for sample in tqdm(samples):
            sample_english = sample
            sample = self.correct_grammar(sample)
            
            encoding = self._tokenizer(sample, return_tensors="pt")
            #encoding = self._tokenizer.encode_plus(
            #    sample,
            #    add_special_tokens=True,
            #    truncation = True,
            #    padding = "max_length",
            #    return_attention_mask = True,
            #    return_tensors = "pt"
            #)
            num_tokens = encoding.input_ids.shape[1]
            
            input_ids = encoding.input_ids.to(self.device)
            target_ids = input_ids.clone().to(self.device)
            
            #print(input_ids, input_ids.shape)
            
            nlls = []
            for mask_idx in range(input_ids[0].shape[0]):
                masked_input_ids = input_ids.clone().to(self.device)
                masked_input_ids[0][mask_idx] = self._tokenizer.mask_token_id
                
                with torch.no_grad():
                    outputs = self._model(input_ids).logits[:,mask_idx,:]
                    softmax = F.softmax(outputs, -1)
                    
                    target_id = target_ids[0][mask_idx]
                    nlls.append(softmax[0,target_id])

            perplexity = torch.exp((-1/num_tokens)*torch.log(torch.stack(nlls)).sum())
            if self._verbose >= 2:
                print(f"[{self._name}] Sample <{sample_english}> has perplexity [{perplexity}]")
            perplexities.append(perplexity.item())

        if self._verbose >= 1:
            print(f"[{self._name}] Final average perplexity: {sum(perplexities)/len(perplexities)}")

        return perplexities 