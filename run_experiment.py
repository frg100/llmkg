import argparse
import torch

import graphs
import models
import utils


def main(args):
    # Load the graph
    graph = None
    if args.verbose >= 1:
        print('[EXPERIMENT] Loading graph')
    if args.graph == 'FB15k237':
        graph = graphs.FB15k237(
            base_path='./data/FB15k-237',
            splits=['train', 'valid','test'],
            verbose=args.verbose
        )
    elif args.graph == 'WN18RR':
        graph = graphs.WN18RR(
            base_path='./data/WN18RR/text',
            splits=['train', 'valid','test'],
            verbose=args.verbose
        )
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.verbose >= 2:
        print('Using device:', device)

    # Load model
    model = FileNotFoundError
    if args.verbose >= 1:
        print('[EXPERIMENT] Loading model')
    if args.verbose >= 2:
            print(f'Loading a model in the [{args.family}] family')
    if args.family == 'GPT2':
        assert args.model in ['gpt2', 'gpt2-large'], 'Invalid model-family pair'
        model = models.GPT2(model_id=args.model, device=device, verbose=args.verbose)
            
    elif args.family == 'BERT':
        model = models.BERT(device=device, verbose=args.verbose)

    # Run experiment
    if args.verbose >= 1:
        print('[EXPERIMENT] Running experiment')
    model.experiment(graph, args.runs, args.samples)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a model-graph discriminatory power experiment.')
    parser.add_argument('--family', type=str, required=True,
                        help='The model family', choices=['GPT2', 'BERT'])
    parser.add_argument('--model', type=str,
                        help='The model type',
                        choices=['gpt2', 'gpt2-large'])

    parser.add_argument('--graph', type=str, required=True,
                        help='The graph', choices=['FB15k237', 'WN18RR'])

    parser.add_argument('--runs', type=int, required=True,
                        help='The number of experiment runs')
    parser.add_argument('--samples', type=int, required=True,
                        help='The number of samples per run')

    parser.add_argument('--verbose', type=int, required=True,
                        help='The level of verbosity')

    args = parser.parse_args()
    main(args)
