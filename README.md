# WakeWord-Detection
Training and evaluation scripts for wake word detection DNN models.

# Requirements
* Tensorflow 2
* PyTorch

>** see requirements.txt for specific packages and version

# Demo

- CRNN
  > python demo.py --models_dir tf_lite_models/CRNN --model_type CRNN
- Wavenet
  > python demo.py --models_dir tf_lite_models/Wavenet --model_type Wavenet
  
  
# Dataset 
["hey-snips" research dataset](https://github.com/sonos/keyword-spotting-research-datasets) 

- Preprocess Original Hey-Snips Dataset:
  
    > python utils/preprocess_dataset.py --data_dir \<path to original dataset> --out_dir \<path to save updated data>
    
    * The purpose of this module is to create a refined version of the original 'hey snips' dataset from Sonos. The first step is to use a VAD to identify the speech content in the signal, and remove silence from the onset and offset. This will be done for all splits, train, test, and development. The script assumes path to the original dataset conforms to the directory structure provided by the dataset publishers. The second step then takes the new train samples and, for each positive sample, selects a segment corresponding to 40-60% of the offset of the original signal, and replaces it with silence or speech from a negative sample. These are then saved out into a separate folder in the 'out_dir' directory called 'enhanced_train_negative'.
    
- Create Datasets for Training, Validation and Testing as H5 vectors:   
  
    > python utils/filter_dataset_to_h5.py --data_dir \<path to dataset> --models_dir \<path to filter.tflite> 
    
    * The purpose of this module is to convert all time domain audio in the datasets to mel-filter banks and save the training vectors and associated labels to H5    files for efficient loading during training.
    

# Models

- Convolutional Recurrent Neural Network (CRNN) described in  Arik *et al.* ["paper"](https://arxiv.org/abs/1703.05390)

  Train CRNN, evaluate metrics, and output tflite models:
  
    > python wwdetect/CRNN/train.py --data_dir \<path to dataset>
    
    * Script to run CRNN training, automatically output both standard model files and tflite files. Supplies basic metrics for final model on test data. Model hyperparameters can be adjusted via variables at top of file. 

- Wavenet described in Coucke *et al.* ["paper"](https://arxiv.org/abs/1811.07684)
    
  Train Wavenet Model:
  
    > python wwdetect/wavenet/train_wavenet.py
    
   Evaluate Wavenet Model:
  
    > python wwdetect/wavenet/evaluate_wavenet.py
   
   Convert Wavenet Model to TF-Lite models (encode and detect):
  
    > python wwdetect/wavenet/convert_wavenet_tflite.py
   
# Evaluation

 - Evaluate models using FAR/FRR:
  
    > python utils/evaluate_models.py --model_type \<CRNN or Wavenet> --models_dir \<path to tflite models> --data_dir \<path to hey snips dataset> 
    
    * The purpose of this module is to evaluate the false rejection rate and false alarms per hour of the given model
    
    > python utils/plot_eval_models.py --results_dir_wavenet \<path to Wavenet tflite models> --results_dir_crnn \<path to CRNN tflite models>
    
    * The purpose of this module is to compare the evaluation results of both models with the described results in the Coucke *et al.* paper
    
## License
all code Copyright 2021: Alireza Bayestehtashk, Amie Roten, Merlin Carson, Meysam Asagari     
except spokestack Copyright 2020: Spokestack, Inc.     
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
