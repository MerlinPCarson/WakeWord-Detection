# WakeWord-Detection
Training and evaluation scripts for wake word detection DNN models.

# Requirements
* Tensorflow 2

# Dataset 
["hey-snips" research dataset](https://github.com/sonos/keyword-spotting-research-datasets) 

# Models
- Convolutional Recurrent Neural Network (CRNN) described in  Arik *et al.* ["paper"](https://arxiv.org/abs/1703.05390)

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
