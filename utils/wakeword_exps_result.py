import os
import sys
import time
import pickle
import argparse
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from glob import glob


def experiment_wavenet(exp_file, exp_type):
    # load experiment model names from experiment metadata 
    exps = pickle.load(open(exp_file,'rb'))
    base_name = os.path.join(os.path.dirname(exp_file), exps['model_name'])
    eval_models = [f'{base_name}_{float(exp):.2f}' for exp in exps['keep_ratios']]

    if exp_type == 'Wakewords':
        return eval_models, exps['num_wakeword']
    elif exp_type == 'Speakers':
        # reverse lists since they are in opposite order as wakeword exps
        return eval_models[::-1], exps['num_speakers'][::-1]

def load_wavenet_results(eval_models):

    # load the results of evaluation from each model in experiment
    results = {}
    for model in eval_models:
        model_results = pickle.load(open(f'{model}_eval_results.npy','rb'))
        results[os.path.basename(model)] = model_results

    return results

def load_CRNN_results(exp_file, exp_type):
    # load experiment results from file
    exps = pickle.load(open(exp_file,'rb'))

    results = {}

    for key, value in reversed(exps.items()):
        stats = {'accuracy': value['stats']['balanced accuracy'], 'false_negative': value['stats']['fn'],
                 'false_positive': value['stats']['fp'], 'precision': value['stats']['precision'],
                 'recall': value['stats']['recall'], 'true_negative': value['stats']['tn'],
                 'true_positive': value['stats']['tp']}
        results[key] = stats

    if exp_type == 'Wakewords':
        exp_vals = [exp['num_train_wakewords_kept'] for _, exp in exps.items()]
    elif exp_type == 'Speakers':
        exp_vals = [exp['num_train_speakers'] for _, exp in exps.items()]

    return results, exp_vals[::-1]

def process_results(model_results, exp_vals):

    assert len(model_results) == len(exp_vals), 'Size mismatch between experiments and model results!'

    # create parallel lists of accurices and number of wakewords in training set
    results = {'accuracy': [], 'exp_val': [], 'f1': []}
    for (model, result), exp_val in zip(model_results.items(), exp_vals):
        results['accuracy'].append(result['accuracy'])
        if np.isnan(result['recall']): 
            results['f1'].append(0)
        else:
            results['f1'].append(2*result['precision']*result['recall']/(result['precision']+result['recall']))
        results['exp_val'].append(str(exp_val))

    return results

def colate_results(exps):

     results = np.array([accs for accs in exps['accuracy']])
     results_f1 = np.array([f1 for f1 in exps['f1']])

     processed_results = {'accuracy': [], 'std_dev': [], 'f1': [], 'f1_std_dev': []}
     
     for col in range(results.shape[1]):
         processed_results['accuracy'].append(np.mean(results[:,col]))
         processed_results['std_dev'].append(np.std(results[:,col]))
         processed_results['f1'].append(np.mean(results_f1[:,col]))
         processed_results['f1_std_dev'].append(np.std(results_f1[:,col]))

     return processed_results

def gather_wavenet_results(args):

    wavenet_exps = {'accuracy': [], 'exp_val': [], 'f1': []} 

    for dir in glob(os.path.join(args.wavenet_eval_dir, '*' + os.path.sep)):

        # load experiment model names from experiment metadata 
        eval_models, exp_vals = experiment_wavenet(os.path.join(dir, args.wavenet_eval_file), args.exp_type)

        # load evaluation results for each model in experiment
        model_results = load_wavenet_results(eval_models)

        # get experiment results given each model's results
        exp_results = process_results(model_results, exp_vals)

        wavenet_exps['accuracy'].append(exp_results['accuracy'])
        wavenet_exps['f1'].append(exp_results['f1'])
        wavenet_exps['exp_val'].append(exp_results['exp_val'])

    results = colate_results(wavenet_exps)

    return results, exp_vals

def gather_CRNN_results(args):

    CRNN_exps = {'accuracy': [], 'exp_val': [], 'f1': []} 

    for exp_file in glob(os.path.join(args.CRNN_eval_dir, '*.npy')):

        # load experiment model names from experiment metadata 
#        eval_models, exp_vals = experiment_CRNN(exp_file, args.exp_type)

        # load evaluation results for each model in experiment
        model_results, exp_vals = load_CRNN_results(exp_file, args.exp_type)

        # get experiment results given each model's results
        exp_results = process_results(model_results, exp_vals)

        CRNN_exps['accuracy'].append(exp_results['accuracy'])
        CRNN_exps['f1'].append(exp_results['f1'])
        CRNN_exps['exp_val'].append(exp_results['exp_val'])

    results = colate_results(CRNN_exps)

    return results, exp_vals

def plot_wakeword_pruning_exps(exp_results, exp_vals, model_type, exp_type, metric):

    # create figure 
    fig, ax = plt.subplots(1,1)

    # reformat ints to strings for x-axis of bar chart
    exp_vals = [str(nm) for nm in exp_vals]

    # prune outlier data
    exp_vals = exp_vals[1:]
    if metric == 'Accuracy':
        y_vals = exp_results['accuracy'][1:]
        y_stds = exp_results['std_dev'][1:]
    elif metric == 'F1-Score':
        y_vals = exp_results['f1'][1:]
        y_stds = exp_results['f1_std_dev'][1:]


    # plot accuracy by number of wakewords in training set
    plt.bar(exp_vals, y_vals, yerr=y_stds, 
            capsize=5.0, color='red', hatch='//', zorder=3, label=model_type)

    # set a tight bound between minimum and maximum accuracies
    plt.ylim((max(0, min(y_vals)-0.005), min(1, max(y_vals)+0.005)))

    # set style
    plt.ylabel(f'{metric}')
    plt.xlabel(f'{exp_type} in Training Set')
    ax.set_facecolor('lightgray')
    plt.grid(color='white', zorder=0)
    plt.tight_layout()
    plt.legend()

    # show the graph 
    plt.show()

def plot_compare_pruning_exps(wavenet_results, CRNN_results, exp_vals, exp_type, metric):

    # create figure 
    fig, ax = plt.subplots(1,1)

    # reformat ints to strings for x-axis of bar chart
    exp_vals = [str(nm) for nm in exp_vals]

    # prune outlier data
    exp_vals = exp_vals[1:]
    if metric == 'Accuracy':
        y_vals_wavenet = wavenet_results['accuracy'][1:]
        y_stds_wavenet = wavenet_results['std_dev'][1:]
        y_vals_CRNN = CRNN_results['accuracy'][1:]
        y_stds_CRNN = CRNN_results['std_dev'][1:]
    elif metric == 'F1-Score':
        y_vals_wavenet = wavenet_results['f1'][1:]
        y_stds_wavenet = wavenet_results['f1_std_dev'][1:]
        y_vals_CRNN = CRNN_results['f1'][1:]
        y_stds_CRNN = CRNN_results['f1_std_dev'][1:]

    locs = np.arange(len(exp_vals)) # the x locations for the groups
    width = 0.35                    # width of the bars

    # plot accuracy by number by experiment pruning
    plt.bar(locs, y_vals_wavenet, width, yerr=y_stds_wavenet, 
            capsize=5.0, color='red', hatch='//', zorder=3, label='Wavenet')

    plt.bar(locs+width, y_vals_CRNN, width, yerr=y_stds_CRNN, 
            capsize=5.0, color='blue', hatch='\\', zorder=3, label='CRNN')

    # set a tight bound between minimum and maximum accuracies
    plt.ylim(min(max(0,min(y_vals_wavenet)-0.005),max(0, min(y_vals_CRNN)-0.005)), max(min(1, max(y_vals_wavenet)+0.005), min(1, max(y_vals_CRNN)+0.005)))
    #plt.ylim((max(0, min(y_vals)-0.005), min(1, max(y_vals)+0.005)))
    plt.xticks(locs + width/2, exp_vals)
    # set style
    plt.ylabel(f'{metric}')
    plt.xlabel(f'{exp_type} in Training Set')
    ax.set_facecolor('lightgray')
    plt.grid(color='white', zorder=0)
    plt.tight_layout()
    plt.legend()

    # save figure
    plt.savefig(f'Exp-{exp_type}-{metric}.pdf')
    # show the graph 
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Wavenet evaluate results of wakeword pruning experiments.')
    parser.add_argument('--CRNN_eval_dir', type=str, default='wwdetect/CRNN/wakeword_exps', 
                                    help='Location of directory with experiments as subdirectories')
    parser.add_argument('--wavenet_eval_dir', type=str, default='wwdetect/wavenet/wakeword_exps', 
                                    help='Location of directory with experiments as subdirectories')
    parser.add_argument('--wavenet_eval_file', type=str, default='wavenet-exps.npy', 
                                    help='Location of file that contains experiment metadata')
    parser.add_argument('--model_type', type=str, default='Wavenet', choices=['CRNN', 'Wavenet'], help='Model type being evaluated.')
    parser.add_argument('--exp_type', type=str, default='Wakewords', choices=['Wakewords', 'Speakers'], help='Model type being evaluated.')
    parser.add_argument('--metric', type=str, default='F1-Score', choices=['Accuracy', 'F1-Score'], help='Model type being evaluated.')

    args = parser.parse_args()
    assert args.wavenet_eval_file, 'No experiment metadata file found,  \
    (--wavenet_eval_file model_path/<model_name>-exps.npy) required!'

    return args

def main(args):
    start = time.time()

    wavenet_results, wavenet_exp_vals = gather_wavenet_results(args)

    CRNN_results, crnn_exp_vals = gather_CRNN_results(args)

    assert np.array_equal(np.array(wavenet_exp_vals), np.array(crnn_exp_vals)), 'Experimental parameters values differ!'

    # plot results for both models
    plot_wakeword_pruning_exps(wavenet_results, wavenet_exp_vals, 'Wavenet', args.exp_type, args.metric)
    plot_wakeword_pruning_exps(CRNN_results, crnn_exp_vals, 'CRNN', args.exp_type, args.metric)

    # compare results for both models
    plot_compare_pruning_exps(wavenet_results, CRNN_results, wavenet_exp_vals, args.exp_type, args.metric)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
