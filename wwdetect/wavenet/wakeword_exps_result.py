import os
import sys
import time
import pickle
import argparse
import matplotlib.pyplot as plt

from evaluate_wavenet import experiment_models


def load_results(eval_models):

    # load the results of evaluation from each model in experiment
    results = {}
    for model in eval_models:
        model_results = pickle.load(open(f'{model}_eval_results.npy','rb'))
        results[os.path.basename(model)] = model_results

    return results

def process_results(model_results, num_wakewords):

    assert len(model_results) == len(num_wakewords), 'Size mismatch between experiments and model results!'

    # create parallel lists of accurices and number of wakewords in training set
    results = {'accuracy': [], 'num_wakeword': []}
    for (model, result), num_wakeword in zip(model_results.items(), num_wakewords):
        results['accuracy'].append(result['accuracy'])
        results['num_wakeword'].append(str(num_wakeword))

    return results

def plot_wakeword_pruning_exp(exp_results, model_type):

    # create figure 
    fig, ax = plt.subplots(1,1)

    # plot accuracy by number of wakewords in training set
    plt.bar(exp_results['num_wakeword'], exp_results['accuracy'], color='red', hatch='//', zorder=3, label=model_type)

    # set a tight bound between minimum and maximum accuracies
    plt.ylim((max(0, exp_results['accuracy'][0]-0.01), min(1, exp_results['accuracy'][-1]+0.01)))

    # set style
    plt.ylabel('Balanced Accuracy')
    plt.xlabel('Wakeword Examples in Training Set')
    ax.set_facecolor('lightgray')
    plt.grid(color='white', zorder=0)
    plt.tight_layout()
    plt.legend()

    # show the graph 
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Wavenet evaluate results of wakeword pruning experiments.')
    parser.add_argument('--eval_file', type=str, default=None, 
                                    help='Location of file that contains experiment metadata')
    parser.add_argument('--model_type', type=str, default='Wavenet', choices=['CRNN', 'Wavenet'], help='Model type being evaluated.')

    args = parser.parse_args()
    assert args.eval_file, 'No experiment metadata file found,  \
    (--eval_file model_path/<model_name>-exps.npy) required!'

    return args

def main(args):
    start = time.time()

    # load experiment model names from experiment metadata 
    eval_models, num_wakewords = experiment_models(args.eval_file)

    # load evaluation results for each model in experiment
    model_results = load_results(eval_models)

    # get experiment results given each model's results
    exp_results = process_results(model_results, num_wakewords)

    # plot results
    plot_wakeword_pruning_exp(exp_results, args.model_type)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
