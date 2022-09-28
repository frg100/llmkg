# LLMKG

## Evaluating Language Models for Common Sense Reasoning on Knowledge Graphs using Perplexity distributions

Federico Reyes Gomez 

Stanford University 

Department of Computer Science 

frg100@cs.stanford.edu 

#### Abstract
As they become more ubiquitous and influential, language models (LMs) need new methods of evaluation to provide a robust view of their biases and performance. We present a new framework for creating common sense reasoning datasets from Knowledge Graphs (KGs) and evaluating language models on these datasets using the distributions of sentence-level perplexity of natural language representations of positive and negative KG triples. Evaluations were performed on four LMs: BERT (Base), RoBERTa (Base), GPT2, and GPT2 (Large) on two KGS: FreeBase15k (FB15K-237) and WordNet18 (WN18RR). To measure whether a LM contains knowledge from a KG, we calculate the p-value from an Independent T-Test on the perplexity distributions of positive and negative samples from the KG. Initial results show high but differing levels of discriminatory ability of pretrained LMs from HuggingFace, with p-values below $p = 0.05$ all the way down to an average $p \approx 1\times 10^{-75}$.


#### Paper
https://github.com/frg100/llmkg/blob/main/LLMKG.pdf
