import os
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

def compare_models(samples, model_a, perplexities_a, model_b, perplexities_b):
    f, ax = plt.subplots(figsize=(20,10))
    ax.set(yscale="log")

    model_one_ppx = np.array([ppx.item() for ppx in perplexities_a])
    model_two_ppx = np.array([ppx.item() for ppx in perplexities_b])

    sort_order = np.argsort(model_one_ppx).astype(int)

    sns.lineplot(x=samples, y=model_one_ppx[sort_order], label=model_a)
    sns.lineplot(x=samples, y=model_two_ppx[sort_order], label=model_b)

    plt.xticks(rotation=90)

    plt.legend()
    plt.ylabel('Perplexity')
    plt.title("Comparing perplexity on sample sentences between two models")

    plt.show()
    
    return plt

def compare_triples(model, perplexities_pos, perplexities_neg):
    f, ax = plt.subplots(figsize=(20,10))
    ax.set(xscale="log")

    pos_ppx = np.array([ppx.item() for ppx in perplexities_pos])
    neg_ppx = np.array([ppx.item() for ppx in perplexities_neg])

    sns.distplot(pos_ppx, label="POSITIVE")
    sns.distplot(neg_ppx, label="NEGATIVE")

    plt.xticks(rotation=90)

    plt.legend()
    plt.ylabel('Perplexity Density')
    plt.title(f"Comparing perplexity on positive and negative sentences for {model}")

    plt.show()
    
    return plt

def plot_results(model_graph_pairings, runs, samples):
    data = {
        'values': [],
        'runs': []
    }

    for model, graph in model_graph_pairings:
        filename = f"results_{model}_{graph}_{runs}x{samples}.txt"
        with open(filename, 'r') as f:
            p_vals = [float(x) for x in f.readlines()]
            
            data['values'] += p_vals
            data['runs'] += [f"{model}_{graph}" for i in range(runs)]

            print(f"{model}_{graph}", p_vals)
 
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(20, 20))

    ax.set_xscale('log')
    ax.set_title('Discriminatory Power of various Models on Knowledge Graph Triples')
    ax.set_xlabel('p-value on a T-test between (pseudo) perplexity \nvalues of each model on positive vs negative \n samples from the Knowledge Graph')
    ax.set_ylabel('Model and Graph')

    plt.axvline(0.05, label='p = 0.05')
    #plt.xlim(1e-12,1)

    plt.legend()

    #sns.distplot(data['values'], label=)
    #Plot the orbital period with horizontal boxes
    sns.boxplot(x="values", y="runs", data=data,
                whis=[0, 100], width=.6, palette="vlag")

    
    #Add in points to show each observation
    sns.stripplot(x="values", y="runs", data=data,
               size=4, color=".3", linewidth=0)

    return f
