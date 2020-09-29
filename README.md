# ECG_Nets
baseline model collection of deep learning applied into ECGs. Those baseline models include 1D-ResNet, 1D-DenseNet, 1D-SE_ResNet, 1D-ResNext,1D-SE_ResNetV2, 1D-SE_ResNext and 1D-Top1Net(the champion model in Tianchi competition). 

The F1-scores of those baseline models are in range of 0.83-0.90. you could fine-tune the parameters to reach a better level.

Step 1:
	downloading datasets. The datasets from Tianchi competition includes dataset A and dataset B. The dataset A consists of 24106 ECGs, of which each has 8 leading records(I，II，V1，V2，V3，V4，V5, V6). while the dataset B consists of 20036 ECGs. Dataset can be downloaded from website. 
	https://pan.baidu.com/s/1fmCuV5i9oifnUNOsFhV0sA  pwd: 8hs2

	Please put the datasets into the fold: all_data. 
	Finally, unzip the data packages.
	
step 2: build environment
	python 3.7 is required. And the requirements can be used directly.
	
	pip install -r requirements.txt

step 3: create csv file. 
	config the dataset path in the data_preparing.py. Then 
	
	python data_preparing.py
	
step 4: choose a baseline model you like.
	7 models are presented in the models fold and configs fold. Those
	
	python main.py --config configs/ResNet50.yaml
	
	if you want ensemble, please config the file configs/ensemble.
	python ensemble.py --config configs/ensemble.yaml

# Reference:
https://tianchi.aliyun.com/competition/entrance/231754/information
https://www.hindawi.com/journals/cmmm/2019/7095137/
https://ieeexplore.ieee.org/abstract/document/9113436
	
	
