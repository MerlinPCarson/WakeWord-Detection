import os
import sys
import time
import pickle
import numpy as np 
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tensorflow.keras.models import load_model

from wavenet_model import build_wavenet_model
from wavenet_loader import HeySnipsDataset 
from train_wavenet import parse_args


def evaluate_model(model, testset):
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

    return results

def evaluate(testset, args):

    # create model and load weights
    model = build_wavenet_model(args)
    model.load_weights(args.eval_model)
    model.summary()

    # evaluate the model
    results = evaluate_model(model, testset)

    # save results to file
    results_file = f'{args.eval_model}_eval_results.npy'
    pickle.dump(results, open(results_file, 'wb'))


def main(args):

    start = time.time()

    # create dataloader
    testset = HeySnipsDataset(os.path.join(args.dataset_dir, args.testset), 
                              num_features=args.num_features, workers=2,
                              shuffle=False)

    print(f'{testset.number_of_examples()} testing examples')

    # evaluate each model in comma seperated list
    eval_models = args.eval_models.split(',')
    for model in eval_models:
        print(f'Evaluating model {model}')
        args.eval_model = model
        evaluate(testset, args)

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))

