import utils

f = utils.plot_results([("BERT_bert-base-cased", "FB15k-237"),("BERT_word_norm", "FB15k-237"),('BERT_roberta-base', 'FB15k-237'), ('GPT2_gpt2', 'FB15k-237'), ('GPT2_gpt2-large', 'FB15k-237')], 25, 1000)
f.set_tight_layout(True)
f.savefig('results_fb_full.png', bbox_inches="tight")
