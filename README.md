# Plant-seedling-classifcation

This work concerns image classification. it's a kaggle competition https://www.kaggle.com/competitions/plant-seedlings-classification/overview.

Plant seedling classification is multiclass image problem. We have 12 different label or type of plant seedling. The aim is to find a model that classify plant seedlings accurately. 
The notebook Plantseedling classification-colab shows the different steps that we proceeded with in order to come up with 3 differents models that I named : custom cnn model for raw data,custom cnn model for processed data, inception model. I saved all three of this models in google drive.


Next, the load_model.py script works as a sort of deployment. If we have a server which would host this script. All we would need to do is to mount google drive account locally on that server. The script will be able to load the models and make use of them to classify, also the image should also be loaded into the server.
