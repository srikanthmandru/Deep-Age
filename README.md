# Title : Deep Age: Towards better approaches for predicting age from images - ALResnet

## Authors: Srikanth Babu Mandru

### Summary:
This project deals with estimating Age from the images. In machine learning terms, it is single-image, multi-class classification problem.

I found this problem interesting since age is one of most significant features that is useful many aspects related to face recognition.

**Dataset source :** https://susanqq.github.io/UTKFace/ 

About data and output classes: 

1. There are initially 1-116 ages people images with around 23,000 images in total. 

2. Ages have been grouped into 14 classes each of 5 consecutive years for first 13 classes and 14th class of pertaining to 66+ age.
            
3. Since the data is not hugely available, ages are grouped to only 8 classes for AL-Resnet Model

### Approach:

**Preprocess of Images:**
- All the images were taken as 256 x 256 x 3 input shape. For few models, images were rezised particularly to fit for those models.
- Pre-processed images using "preprocess_input" function from imagenet utils module from keras for all models except Resnet50, AL-Resnet and InceptionV3.  
- For remaining models (not preprocessed in above point), pre-processed images using normalization technique (divide by 255) to standardize pixel values of images. 
- Fit Generator function has been used for few models but haven't noticed any improvement in model performance. This might be due to losing the age sensitive features from images by preprocessing using fit generator.

**Models and Evaluation:**

• Implemented Softmax Classifier, Deep Neural Networks, Convolutional Neural Networks(CNN) and RNNs using Keras and TensorFlow packages for python.

• Implemented AL-Resnet architecture with Resnet50V2 as global model.

• Evaluated models through accuracy as metric. Over all models, AL-Resnet has achieved high accuracy on test dataset.

#### Results: 

Evaluated models using accuracy, precision, recall, F1 score, ROC and AUC metrics and achieved ‘43.16%’ accuracy on test data with 14 classes and ‘59.04%’ with 8 classes. Furthermore, observed that models are doing well on predicting the age group of kids and old-aged people, showing coherence in performance with human prediction levels.

**References:**

[1] Gil Levi and Tal Hassner. Age and Gender Classification Using Convolutional Neural Networks. IEEE Workshop on Analysis and Modeling of Faces and Gestures (AMFG), at the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, 2015.

[2] K. Zhang, N. Liu, X. Yuan, X. Guo, C. Gao, and Z. Zhao. Fine-grained age estimation in the wild with attention LSTMnetworks. arXiv preprint arXiv:1805.10445v2, 2019.

[3] Karen Simonyan, Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. arXiv preprint arXiv:1409.1556v6, 2015.

[4] Krizhevsky, Alex; Sutskever, Ilya; Hinton, Geoffrey E. (2017-05-24). "ImageNet classification with deep convolutional neural networks" (PDF). Communications of the ACM. 60 (6): 84–90. doi:10.1145/3065386. ISSN 0001- 0782.

[5] LeCun, Yann; Léon Bottou; Yoshua Bengio; Patrick Haffner (1998). "Gradient-based learning applied to document recognition" (PDF). Proceedings of the IEEE. 86 (11): 2278–2324. CiteSeerX 10.1.1.32.9552. doi:10.1109/5.726791. Retrieved October 7, 2016.

[6] A. Ekmekji, "Convolutional neural networks for age and gender classification", 2016.

[7] https://keras.io

**Final comments:**

I will try to tune existing models further with other available datasets for age estimation (this eliminates the overfitting problem) and also implement other models to get better accuracy so that model would generalize well to unseen real-world data. Stay tuned!!!

**Descriptions of the functionality of each file that is included in this repository are given below :**

- "Deep_Age_prediction_Read_files.ipynb" file is used for reading the data from raw images and convert to hdf5 format for storing

- "Deep_Age_Classifiers_data_split.ipynb " file makes the data split and some label preprocessing and resultant training and test data is stored as hdf5 data.

- " Deep_Age_Classifiers_v2.ipynb " file comprises of training softmax, VGG19 , InceptionV3, 4C2F-NN models. 

- For 4C2F-NN, architecture is implemented in " research_paper.py " file and is loaded into " Deep_Age_Classifiers_v2.ipynb " 
        while training.

- " Alexnet_Lenet.ipynb " file is used for training ALexnet and Lenet models

- " Resnet_Model.ipynb " file is used for training the Resnet50V2 model

- " AL_resnet50V2.ipynb " file used for training the AL-Resnet model

- "AL_resnet50V2-8classes.ipynb" file uses 8 classes for AL-Resnet Model Training.

File usage instructions : 

1. All the files have code to save the models and their weights. 

2. "train_func" function in files is used to train the models using either with data augmentation or without augmentation. They are set to not use data augmentation by using parameter "data_augmentation" = False. Using Data augmentation, does not show any improvements so better to train without augmentation. However, code for using fit_generator is also included. 

**Tools and APIs:**

Tensorflow, Keras, scikit-learn.
