# Blur-image-classification

**Predict an image which is blur or not(Blue detection of an image)**


Blue images can be detected with the help of Laplacian operators by taking the maximum value. If that value is below a threshold value, then the image can be considered as blurred otherwise not. 

Description of the dataset: 
**dataset link:** https://mklab.iti.gr/results/certh-image-blur-dataset/
The Training Set consists of:
		630 undistorted (clear) images
		220 naturally-blurred images
		150 artificially-distorted images

The Evaluation Set consists of two individual data sets :
	The Natural Blur Set which consists of:
		589 undistorted (clear) images
		411 naturally-blurred images
		
	The Digital Blur Set
		30 undistorted (clear) images
		450 artificially-blurred images

Steps:
1. create a virtual environment 


2. make  sure all libraries are installed


3. Training part : 
            3.1 -  train the training dataset and load their weights using pickle 
            3.2  - run the training set on terminal
                     python training_dataset.py


4. Testing Part :
            4.1 - load the weights of evaluation dataset using  pickle 
          and run the file using command
                     python evaluation_dataset.py


5. Predict part : 
                     python main.py
