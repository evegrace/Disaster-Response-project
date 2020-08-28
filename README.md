## Disaster Response Web App

### Installation
The code contained in this repository was written in HTML and Python 3, and requires the following 
Python packages: json, plotly, pandas,numpy, nltk, flask, sklearn, sqlalchemy, re, pickle.

### Project Overview
This project contains namely three processes, that is ETL(Extract, Transform, Load) and then creating a Machine Learning Pipeline, 
saving the file as a pickle and finally creating a web app that can be used during a disaster(such as an earthquake, hurricane, etc) 
for faster response from agencies.

The app uses a ML model to categorize any new messages received, and the repository also contains the code used to train the model and to prepare any new datasets for model training purposes.

### File Descriptions
**process_data.py**: This code as named processes the data, in this case our input was csv files containing message data and message categories (labels), and creates 
an SQLite database containing a merged and cleaned version of this data.

**train_classifier.py**: This code takes the SQLite database produced by process_data.py as an input and uses the data 
contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.

**ETL Pipeline Preparation.ipynb**
The code and analysis contained in this Jupyter notebook was used in the development of process_data.py.


**ML Pipeline Preparation.ipynb** 
The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. 

**data**
 This folder contains sample messages and categories datasets in csv format.

**app**
 This folder contains all of the files necessary to run and render the web app.

**images**
Folder contains screenshots of the results.

**Summary of the Results**
Below are screenshots of the results obtained.
![capture1](https://user-images.githubusercontent.com/68501659/91150038-85251580-e6c4-11ea-9439-2024d3298583.jpg)
![capture2](https://user-images.githubusercontent.com/68501659/91150045-87876f80-e6c4-11ea-8fce-4707c53be35d.jpg)

### Warning
Care should be taken if relying on the results of this app for decision making purposes.

The datasets included in this repository are very unbalanced, with very few positive examples for several message categories.
In some cases, the proportion of positive examples is less than 5%, or even less than 1%. In such cases, even though the classifier accuracy 
is very high (since it tends to predict that the message does not fall into these categories), the classifier recall 
(i.e. the proportion of positive examples that were correctly labelled) tends to be very low. 

### Licensing, Authors, Acknowledgements
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity. 
The data was originally sourced by Udacity from Figure Eight.
