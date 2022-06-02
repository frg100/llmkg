import utils

f = utils.plot_results([('BERT', 'FB15k-237'), ('GPT2_gpt2', 'FB15k-237'), ('GPT2_gpt2-large', 'FB15k-237')], 25, 1000)
f.savefig('results_1_long.png')
