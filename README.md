# WakeWord-Detection
Training and evaluation scripts for wake word detection DNN models.

# Requirements
* Tensorflow 2

# Dataset 
["hey-snips" research dataset](https://github.com/sonos/keyword-spotting-research-datasets) 

# Models
- Convolutional Recurrent Neural Network (CRNN) described in  Arik *et al.* ["paper"](https://arxiv.org/abs/1703.05390)
  
  Preprocess Original Hey-Snips Dataset:
  
    > python utils/preprocess_dataset.py --data_dir <path to original dataset> --out_dir <path to save updated data>
    
    * The purpose of this module is to create a refined version of the original 'hey snips' dataset from Sonos. The first step is to use a VAD to identify the speech content in the signal, and remove silence from the onset and offset. This will be done for all splits, train, test, and development. The script assumes path to the original dataset conforms to the directory structure provided by the dataset publishers. The second step then takes the new train samples and, for each positive sample, selects a segment corresponding to 40-60% of the offset of the original signal, and replaces it with silence or speech from a negative sample. These are then saved out into a separate folder in the 'out_dir' directory called 'enhanced_train_negative'.

  Create Datasets as H5 Vectors:
  
    * The same script as below can be used to employ the filter model to extract features from the dataset and output as H5 files, for use in model training.

  Train CRNN, evaluate metrics, and output tflite models:
  
    > python wwdetect/CRNN/train.py --data_dir <path to dataset>
    
    * Script to run CRNN training, automatically output both standard model files and tflite files. Supplies basic metrics for final model on test data. Model hyperparameters can be adjusted via variables at top of file. 

  Evaluate model using FAR/FRR:
  
    > python utils/evaluate_models.py --model_type CRNN --models_dir <path to tflite models> --data_dir <path to hey snips dataset> 

- Wavenet described in Coucke *et al.* ["paper"](https://arxiv.org/abs/1811.07684)

  Create Datasets for Training, Validation and Testing as H5 vectors:   
  
    > python utils/filter_dataset_to_h5.py --data_dir \<path to dataset> --models_dir \<path to filter.tflite> 
    
  Train Wavenet Model:
  
    > python wwdetect/wavenet/train_wavenet.py
    
   Evaluate Wavenet Model:
  
    > python wwdetect/wavenet/evaluate_wavenet.py
   
   Convert Wavenet Model to TF-Lite models (encode and detect):
  
    > python wwdetect/wavenet/convert_wavenet_tflite.py
    
## License
Copyright 2021: Alireza Bayestehtashk, Amie Roten, Merlin Carson, Meysam Asagari  
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
