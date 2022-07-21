# Version  of solution for "TGS Salt Identification Challenge" competition 

link to TGS competition: https://www.kaggle.com/c/tgs-salt-identification-challenge 


Folders:
 /train
   /images  - TRAIN Data: source train images (Data should be downloaded from kaggle)
   /masks   - TRAIN Data: source masks of salt (Data should be downloaded from kaggle)
 /images    - TEST Data: images without masks, for masks prediction
 /models    - for store models weights (as .csv) and information of models performance (as .csv.html)
 /results   - for store prediction results for TEST Data (images from /images)
 
 depths.csv             - additional Data-file, should be gotten from kaggle
 main.py                - main entry point for training stage: training model based on Train data
 processing_module.py   - library with traingng logic
 opt_utils.py           - utils for NN
 
 predict.py             - main entry point for prediction stage: predicting masks based on Test data
 
 predict-ans.ipynb      - megre several prediction results