import os
import sys
import time
import pickle
import numpy as np 
from tqdm import tqdm
from zipfile import ZipFile

from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tensorflow.keras.models import load_model

from wavenet_model import build_wavenet_model
from wavenet_loader import HeySnipsDataset 
from train_wavenet import parse_args


def experiment_models(exp_file):
    # load experiment model names from experiment metadata 
    exps = pickle.load(open(exp_file,'rb'))
    base_name = os.path.join(os.path.dirname(exp_file), exps['model_name'])
    eval_models = [f'{base_name}_{exp}' for exp in exps['keep_ratios']]
    return eval_models, exps['num_wakeword']

def evaluate_model(model, testset, zip_missed=False):
    preds = model.predict(testset, verbose=1)
    targets = testset.get_labels()

    class_preds = np.argmax(preds, axis=1) 
       
    # since size of dataset might not be a multiple of batchsize
    # only take as many targets as there are predictions
    cf = confusion_matrix(targets[:len(class_preds)], class_preds)
    print(f'\nConfusion matrix:\n {cf}')

    # get metrics
    tn, fp, fn, tp = cf.ravel()
    results = {'true_negative': tn, 'false_positive': fp, 
               'true_positive': tp, 'false_negative': fn,
               'recall': tp/(tp+fp), 'precision': tp/(tp+fn),
               'accuracy': balanced_accuracy_score(targets[:len(class_preds)], class_preds)}

    # print all metrics in results
    for key, val in results.items():
        print(f'{key}: {val}')

    if zip_missed:
        # get file names for incorrect predictions
        fp_files, fn_files = misclassified_audio_files(targets[:len(class_preds)], 
                                                       class_preds, testset.get_filenames())
    
        # zip raw audio files for incorrect predictions
        zip_files(fp_files, args.audio_dir, f'{args.eval_model}_fp_wavs.zip')
        zip_files(fn_files, args.audio_dir, f'{args.eval_model}_fn_wavs.zip')

    return results

def evaluate(testset, args):

    # create model and load weights
    model = build_wavenet_model(args)
    model.load_weights(args.eval_model)

    # evaluate the model
    results = evaluate_model(model, testset, zip_missed=args.zip_missed)

    # save results to file
    results_file = f'{args.eval_model}_eval_results.npy'
    pickle.dump(results, open(results_file, 'wb'))

def misclassified_audio_files(targets, preds, file_names):

    assert len(targets) == len(preds) == len(file_names), \
                'mismatch between lengths of targets, preds, file names'

    fp_files = [f'{file_names[i]}.wav' for i, val in enumerate(targets)
                if preds[i] == 1 and val == 0]

    fn_files = [f'{file_names[i]}.wav' for i, val in enumerate(targets)
                if preds[i] == 0 and val == 1]

    return fp_files, fn_files

def zip_files(wav_files, audio_dir, outfile):

    with ZipFile(outfile, 'w') as zip:
        for wav in wav_files:
            zip.write(os.path.join(audio_dir, wav), 
                      os.path.join(os.path.basename(outfile.replace('.zip','')), wav))


def main(args):

    start = time.time()

    # create dataloader
    testset = HeySnipsDataset([os.path.join(args.dataset_dir, args.testset)], 
                              num_features=args.num_features, workers=2,
                              shuffle=False)

    print(f'{testset.number_of_examples()} testing examples')

    if not args.eval_file:
        # evaluate each model in comma seperated list
        eval_models = args.eval_model.split(',')
    else:
        # load model names from experiment metadata file
        eval_models, _ = experiment_models(args.eval_file)

    for model in eval_models:
        print(f'Evaluating model {model}')
        args.eval_model = model
        evaluate(testset, args)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

