This file (project) is the source code of our proposed model.
(1) Download the datasets at this url:
### https://drive.google.com/drive/folders/1UNqQEb6h9_DcFEMTInjNsuaA-6HdzqEF?usp=share_link
(2) unzip preprocessedDataset.zip
(3) copy the folder "preprocessedDataset/beijing" and all files in this folder, to directory "GTCPredictor/resource/preprocess" in this project "GTCPredictor".
(4) copy the folder "preprocessedDataset/xi_an" and all files in this folder, to directory "GTCPredictor/resource/preprocess" in this project "GTCPredictor".
(5) Open this project with pycharm.
(6) Go to GTCPredictor.GTCP.Invoker.py, and run it. Then a model is trained for Beijing AQI dataset.
(7) Go to GTCPredictor.GTCP_u2.Invoker.py, and run it. Then a model is trained for Beijing PM2.5 dataset.
(8) Go to GTCPredictor.GTCP_xi_an.Invoker.py, and run it. Then a model is trained for Xi'an AQI dataset.
(9) Go to GTCPredictor.GTCP_xi_an.Const.py, and set as follows: 
dataset_name='PM2.5_processed' (Line 5-6)
gaussian_PG_hidden_size=32 (Line 28)
framework_hidden_size=18 (Line 29)
intra_coupling_weight=0.01 (Line 30)
(10) Go to GTCPredictor.GTCP_xi_an.Invoker.py, and run it. Then a model is trained for Xi'an PM2.5 dataset.


p.s. running environment:
python 3.6
pytorch	1.8.0
cuda 10.1
