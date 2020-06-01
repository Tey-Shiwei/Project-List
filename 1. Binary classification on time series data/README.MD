#  CS5228 Final Project, collaborator: Chua Chin Siang and Chan Jun Wei

### Environment
anaconda, python 3.6 

## Requirements
1. numpy==1.17.3
2. scipy==1.3.1
3. wrapt==1.11.2

### Setup
``pip install -r requirements.txt``

### Data
1. Preprocessed train data are saved in ``filtered_features`` directory 
2. Preprocessed test data are saved in ``filtered_test_features`` directory

### Run model
1. To extract features, run ``python generate_ts_fresh_features.py``
2. To run the model, just run ``python submission_ml.py``. 
