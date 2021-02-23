import os
import sys
import time
import pickle

from train_wavenet import parse_args
from evaluate_wavenet import experiment_models


def load_results(eval_models):

    results = {}
    for model in eval_models:
        model_results = pickle.load(open(f'{model}_eval_results.npy','rb'))
        results[os.path.basename(model)] = model_results

    return results

def process_results(model_results, num_wakewords):

    assert len(model_results) == len(num_wakewords), 'Size mismatch between experiments and model results!'

    results = {'accuracy': [], 'num_wakeword': []}
    for (model, result), num_wakeword in zip(model_results.items(), num_wakewords):
        results['accuracy'].append(result['accuracy'])
        results['num_wakeword'].append(num_wakeword)

    return results

def plot_wakeword_pruning_exp(exp_results):
    print(exp_results)

def main(args):
    start = time.time()

    assert args.eval_file, 'No experiment metadata file found,  \
    (--eval_file model_path/<model_name>-exps.npy) required!'

    # load experiment model names from experiment metadata 
    eval_models, num_wakewords = experiment_models(args.eval_file)

    # load evaluation results for each model in experiment
    model_results = load_results(eval_models)

    # get experiment results given each model's results
    exp_results = process_results(model_results, num_wakewords)

    # plot results
    plot_wakeword_pruning_exp(exp_results)


    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
