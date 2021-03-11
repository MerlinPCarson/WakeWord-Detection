import os
import sys
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from glob import glob


def experiment_models(exp_file, exp_type):
    # load experiment model names from experiment metadata 
    exps = pickle.load(open(exp_file,'rb'))
    base_name = os.path.join(os.path.dirname(exp_file), exps['model_name'])
    eval_models = [f'{base_name}_{float(exp):.2f}' for exp in exps['keep_ratios']]

    if exp_type == 'Wakewords':
        return eval_models, exps['num_wakeword']
    elif exp_type == 'Speakers':
        # reverse lists since they are in opposite order as wakeword exps
        return eval_models[::-1], exps['num_speakers'][::-1]

def load_results(eval_models):

    # load the results of evaluation from each model in experiment
    results = {}
    for model in eval_models:
        model_results = pickle.load(open(f'{model}_eval_results.npy','rb'))
        results[os.path.basename(model)] = model_results

    return results

def process_results(model_results, exp_vals):

    assert len(model_results) == len(exp_vals), 'Size mismatch between experiments and model results!'

    # create parallel lists of accurices and number of wakewords in training set
    results = {'accuracy': [], 'exp_val': []}
    for (model, result), exp_val in zip(model_results.items(), exp_vals):
        results['accuracy'].append(result['accuracy'])
        results['exp_val'].append(str(exp_val))

    return results

def colate_results(exps):

     results = np.array([accs for accs in exps['accuracy']])

     processed_results = {'accuracy': [], 'std_dev': []}
     
     for col in range(results.shape[1]):
         processed_results['accuracy'].append(np.mean(results[:,col]))
         processed_results['std_dev'].append(np.std(results[:,col]))

     return processed_results

def plot_wakeword_pruning_exps(exp_results, exp_vals, model_type, exp_type):

    # create figure 
    fig, ax = plt.subplots(1,1)

    # reformat ints to strings for x-axis of bar chart
    exp_vals = [str(nm) for nm in exp_vals]

    # prune outlier data
    exp_results['accuracy'] = exp_results['accuracy'][1:]
    exp_results['std_dev'] = exp_results['std_dev'][1:]
    exp_vals = exp_vals[1:]

    # plot accuracy by number of wakewords in training set
    plt.bar(exp_vals, exp_results['accuracy'], yerr=exp_results['std_dev'], 
            capsize=5.0, color='red', hatch='//', zorder=3, label=model_type)

    # set a tight bound between minimum and maximum accuracies
    plt.ylim((max(0, exp_results['accuracy'][0]-0.01), min(1, exp_results['accuracy'][-1]+0.01)))

    # set style
    plt.ylabel('Balanced Accuracy')
    plt.xlabel(f'{exp_type} in Training Set')
    ax.set_facecolor('lightgray')
    plt.grid(color='white', zorder=0)
    plt.tight_layout()
    plt.legend()

    # show the graph 
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Wavenet evaluate results of wakeword pruning experiments.')
    parser.add_argument('--eval_dir', type=str, default='wwdetect/wavenet/speaker_exps', 
                                    help='Location of directory with experiments as subdirectories')
    parser.add_argument('--eval_file', type=str, default='wavenet-speaker_exps.npy', 
                                    help='Location of file that contains experiment metadata')
    parser.add_argument('--model_type', type=str, default='Wavenet', choices=['CRNN', 'Wavenet'], help='Model type being evaluated.')
    parser.add_argument('--exp_type', type=str, default='Speakers', choices=['Wakewords', 'Speakers'], help='Model type being evaluated.')

    args = parser.parse_args()
    assert args.eval_file, 'No experiment metadata file found,  \
    (--eval_file model_path/<model_name>-exps.npy) required!'

    return args

def main(args):
    start = time.time()

    wakeword_exps = {'accuracy': [], 'exp_val': []} 
      
    for dir in glob(os.path.join(args.eval_dir, '*' + os.path.sep)):

        # load experiment model names from experiment metadata 
        eval_models, exp_vals = experiment_models(os.path.join(dir, args.eval_file), args.exp_type)

        # load evaluation results for each model in experiment
        model_results = load_results(eval_models)

        # get experiment results given each model's results
        exp_results = process_results(model_results, exp_vals)

        wakeword_exps['accuracy'].append(exp_results['accuracy'])
        wakeword_exps['exp_val'].append(exp_results['exp_val'])

#    verify_results(wakeword_exps)

    results = colate_results(wakeword_exps)

    # plot results
    plot_wakeword_pruning_exps(results, exp_vals, args.model_type, args.exp_type)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
