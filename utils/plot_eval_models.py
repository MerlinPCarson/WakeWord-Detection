import os
import sys
import time
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from evaluate_models import testset_files, duration_test

# estimate of Coucke et al. wavenet paper results from figure
WAVENET_PAPER_RESULTS = [[0.0, 0.0, 0.1, 0.2, 0.4, 0.8, 1.0, 4.5],
                         [0.025, 0.015, 0.01, 0.005, 0.0045, 0.004, 0.0, 0.0]]

def plot_FRR_FAR(results):
    
    fig, ax = plt.subplots(1,2)
    ax[0].set_xlabel("Posterior Threshold")
    ax[0].set_ylabel("False Rejection Rate")
    ax[0].set_facecolor('lightgray')
    ax[0].grid(color='white')

    ax[1].set_xlabel("Posterior Threshold")
    ax[1].set_ylabel("False Accepts per Hour")
    ax[1].set_facecolor('lightgray')
    ax[1].grid(color='white')

    for model in results:
        # plot data
        ax[0].plot(results[model]['thresholds'], results[model]['FRR'], label=model)
        ax[1].plot(results[model]['thresholds'], results[model]['FAR'], label=model)

    plt.legend()
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(1,1)
    ax.set_facecolor('lightgray')
    for model in results:
        color = 'red'
        if model == 'CRNN':
            color = 'blue'
        plt.plot(results[model]['smooth_FAR'], results[model]['smooth_FRR'], color=color,
                 label=f"{model} thresholds: {{{results[model]['thresholds'][0]:.2f},..,{results[model]['thresholds'][-1]:.3f}}}")

    # add max values to paper est. results to fit graph
    xmin, xmax, ymin, ymax = plt.axis()
    print(xmax)
    WAVENET_PAPER_RESULTS[0].append(xmax)
    WAVENET_PAPER_RESULTS[1].insert(0, ymax)
    plt.plot(WAVENET_PAPER_RESULTS[0], WAVENET_PAPER_RESULTS[1], color='green', label='Wavenet Coucke et al. Paper')
    plt.xlim(0,WAVENET_PAPER_RESULTS[0][-1])

    plt.xlabel("False Alarms per Hour")
    plt.ylabel("False Rejection Rate")

    # sets for bounds for plot
    plt.xlim(0,12)
#    plt.ylim(0,0.25)

    plt.grid(color='white')
    plt.legend()
    plt.tight_layout()
    plt.savefig('far_frr.pdf')
    plt.show()
    plt.close()

def load_model_preds(model_types, results_dir, smooth_window=30):
    assert len(model_types) == len(results_dir), 'Size mismatch between number of models and results!'

    # datastruct for holding posteriors
    results = {model: {} for model in model_types}

    for model, result_dir in zip(model_types, results_dir):
        results[model]['wakeword'] = pickle.load(open(os.path.join(result_dir, f'{model}_all_wakeword.pkl'),'rb'))
        results[model]['not_wakeword'] = pickle.load(open(os.path.join(result_dir, f'{model}_no_wakeword.pkl'), 'rb'))
        # smooth not keyword posteriors using average filter
        results[model]['smooth_not_wakeword'] =  np.convolve(results[model]['not_wakeword'], np.ones((smooth_window,))/smooth_window, mode='same')

    return results

def threshold_accepts(posteriors, threshold):
    
    prev_wake = False
    accepts = 0

    for posterior in posteriors:
        if posterior > threshold and not prev_wake:
                accepts += 1
        prev_wake = False
        if posterior > threshold:
                prev_wake = True

    return accepts

def process_results(results, num_wakewords, not_wakeword_duration_hrs):

    for model in results:

        # range of thresholds to apply to posteriors
        thresholds = np.arange(0.5,0.9905,0.001)

        FRR = []
        FAR = []
        print(f'Sweeping thresholds over posteriors for {model} model')
        for threshold in tqdm(thresholds):

            # measure true positives
            accepts = (results[model]['wakeword']>threshold).sum()
            rejects = num_wakewords - accepts
            reject_percentage = rejects / num_wakewords
            FRR.append(reject_percentage)

            accepts = 0

            # measure false positives
            accepts = threshold_accepts(results[model]['smooth_not_wakeword'], threshold)
            accepts_rate = accepts / not_wakeword_duration_hrs 
            FAR.append(accepts_rate)

        results[model]['FRR'] = sorted(FRR)[::-1]
        results[model]['FAR'] = sorted(FAR)
        results[model]['smooth_FAR'] =  np.convolve(results[model]['FAR'], np.ones((30,))/30, mode='valid')
        results[model]['smooth_FRR'] =  np.convolve(results[model]['FRR'], np.ones((30,))/30, mode='valid')
        results[model]['thresholds'] = thresholds

    return results

def parse_args():
    parser = argparse.ArgumentParser(description='Plots evaluation results for multiple models.')
    parser.add_argument('--model_types', type=str, default='CRNN,Wavenet', choices=['CRNN,Wavenet'], help='Model types being evaluated.') 
    parser.add_argument('--data_dir', type=str, default='data/hey_snips_research_6k_en_train_eval_clean_ter/',
                        help='Directory with Hey Snips raw dataset')
    parser.add_argument('--results_dir_wavenet', type=str, default='utils/results/wavenet_98.99', help="Directory to load Wavenet results from")
    parser.add_argument('--results_dir_crnn', type=str, default='utils/results/CRNN', help="Directory to load CRNN results from")
    parser.add_argument('--eval_dir', type=str, default='data/evaluation/same_len',
                        help='Directory to save and load concatenated wav files from')
    parser.add_argument('--pos_samples', type=str, default='hey_snips_long.wav', help='File for concatenated positive class samples')
    parser.add_argument('--neg_samples', type=str, default='not_hey_snips_long.wav', help='File for concatenated negative class samples')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate for audio (Hz)')
    parser.add_argument('--frame_width', type=int, default=20, help='Frame width for audio in (ms)')
    args = parser.parse_args()
    return args

def main(args):
    start = time.time()

    # load audio file paths
    wakeword_files, not_wakeword_files = testset_files(args.data_dir)

    # get number of wakeword examples 
    num_wakewords = len(wakeword_files)

    model_types = args.model_types.split(',')

    model_preds = load_model_preds(model_types, [args.results_dir_crnn, args.results_dir_wavenet])

    # create paths to evaluation wav files
    FAR_path = os.path.join(args.eval_dir, args.neg_samples)

    # get duration of not wakeword test audio
    print('Calculating duration of non-wakeword audio')
    not_wakeword_duration_hrs = duration_test(FAR_path, args.sample_rate)/3600
    print(f'Duration of non-wakeword audio is {not_wakeword_duration_hrs:.2f} hrs')

    # get FRR and FAR results for each model
    results = process_results(model_preds, num_wakewords, not_wakeword_duration_hrs)

    # plot the FRR by FAR
    plot_FRR_FAR(results)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
