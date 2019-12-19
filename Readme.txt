

This file describes the functionality of different files that are included in the folder

- "Deep_Age_prediction_Read_files.ipynb" file is used for reading the data from raw images and convert to hdf5 format for storing

- "Deep_Age_Classifiers_data_split.ipynb " file makes the data split and some label preprocessing and resultant training and test data is stored as hdf5 data.

- " Deep_Age_Classifiers_v2.ipynb " file comprises of training softmax, VGG19 , InceptionV3, 4C2F-NN models. 

    - For 4C2F-NN, architecture is implemented in " research_paper.py " file and is loaded into " Deep_Age_Classifiers_v2.ipynb " 
        while training.

- " Alexnet_Lenet.ipynb " file is used for training ALexnet and Lenet models

- " Resnet_Model.ipynb " file is used for training the Resnet50V2 model

- " AL_resnet50V2.ipynb " file used for training the AL-Resnet model

- "AL_resnet50V2-8classes.ipynb" file uses 8 classes for AL-Resnet Model Training. 


Instructions :
Dataset source : https://susanqq.github.io/UTKFace/



All the files have code to save the models and their weights. 

"train_func" function in files is used to train the models using either with data augmentation or without augmentation. They are set to not use data augmentation by using parameter "data_augmentation" = False. Using Data augmentation, does not show any improvements so better to train without augmentation. However, code for using fit_generator is also included. 

Thanks for reading and your interest. 




