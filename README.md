# Hindi MNIST Classification using Water Accumulation method

**Project Overview**
=====================

This project is a simple implementation of a machine learning model for classifying handwritten digits in the Hindi language using the Water Accumulation method. The machine learning model is trained on the features extracted from the input images using the Water Accumulation method. This model is used to predict the correct label for a given input image. This project was inspired by this [video](https://youtu.be/CC4G_xKK2g8?si=MbyYaG73GaxDrL3X) from [PickentCode](https://www.youtube.com/@PickentCode). The code in reference is available [here](https://github.com/PickentCode/KNN-Digit-Recognition). 
I used Python with Streamlit instead of JavaScript to create this project. Other differences being that I used [Hindi MNIST](https://www.kaggle.com/datasets/imbikramsaha/hindi-mnist/data) dataset instead of MNIST dataset and I used other classifiers instead of KNN.

**Key Features**
---------------

* Data loading and preprocessing using Pandas
* Model training using Scikit-learn's SVC and RandomForestClassifier algorithms
* Hyperparameter tuning GridSearchCV was used to find the best model parameters
* Model and Data caching was used to ensure fast and efficient repeated use
* Support for loading and saving model parameters to a files

**Project Structure**
--------------------

* `helper_functions.py` contains helper functions for the project
* `main.py` contains the Streamlit app code
* `create_train_test_df.py` contains code for creating the train and test datasets
* `feature_extractors.py` contains code for extracting features from input images
* `movement_simulators.py` contains code for simulating water pouring and visualization

**Usage**
-----

This is just a demo project and not intended for real-world use. Please use at your own risk. 