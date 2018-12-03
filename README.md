
# Price Classification of Mobile Phones

## Introduction
>* This is data capstone project. The dataset was taken from Kaggle. The full database contains 21 features of mobile phone of varies companies.There are two datasets, one is train.csv(2000 rows and 21 columns), another is test.csv (1000 rows and 21 columns). The aim of this project is “find out the relation between features of mobile phone and using the machine learning technique to predict the right price range of a mobile phone in the competitive mobile phone market”.

## Data Source
> The link of data as following:  
> https://www.kaggle.com/iabhishekofficial/mobile-price-classification/version/1

## Limitation
> The limitation of this dataset is that the dataset only provided us a price range for the mobile phones rather than specific prices.

# Setup
### Platform：
>* I used Anaconda-Juypter( Download from following link:https://www.anaconda.com/download/) to write the code
>* To see the code and results in the terminal following the "Installation" section that I introduce.
### Dependencies: 
Python 3
### Modules：
>* Numpy
>* Math
>* Pandas
>* Seaborn
>* Matplotlib
>* Scikit-learn
>* Torch
>* Missingno
>* Random

# Project Files
#### README.md: Describing the project and how to run the code
#### Data_Science_Capston_Project_Codes_and_Data"
>* train.csv: The dataset that we used build model and test the model performance
>* test.csv: The dataset that we used to predict price range of mobile phone
>* Data_Science_Capston_Project.py: Code in py format
>* Data_Science_Capston_Project.ipynb: Code in ipynb format
#### Data_Science_Capston_Project-Report: 
>* Data_Science_Capston_Project_Report.pdf: The report of our project. I used the results of 'Data_Science_Capston_Project.ipynb' to write this report.
#### Data_Science_Capston_Project_Presentation_Slides: I used the results of 'Data_Science_Capston_Project.ipynb' to mke the slices
>* Data_Science_Capston_Project_Presentation_Slides.ppdx: The presentation slices in ppdx format

## Installation
#### To run the code:

##### For Windows Users: 
* 1) Download git from the following link: https://git-scm.com/download/win and install it by simple click "Next" command
* 2).Go to your Desktop, and right click left-click the mouse, and type, then choose "Git Bush Here
* (Or just simply use Win+R and type 'cmd' to open a cmd windows)
* 3).In the cmd windows, type 'cd ~/Downloads'
* 4).Then, type 'git clone https://github.com/evazhang720/Data_Science_Capston_Project.git'
* 5) Then type 'cd ~/Data_Science_Capston_Project'
* 6). Then type 'cd ~/Data_Science_Capston_Project_Codes_and_Data'
* 7). Then type 'python3 Data_Science_Capston_Project.py' to run the the code in py file in the terminal

##### For Mac Users:
* 1).Open the 'Spotlight Search'
* 2).Type the'terminal' in the 'Spotlight Search'
* 3).In the terminal, type 'cd Downloads'
* 4). Then, type 'git clone https://github.com/evazhang720/Data_Science_Capston_Project.git''
* 5). Then type 'cd Data_Science_Capston_Project'
* 6). Then type 'Data_Science_Capston_Project_Codes_and_Data'
* 7). Then type 'python3 Data_Science_Capston_Project.py' to run the the code in py file in the terminal

## Goal
>* In this project, I will apply machine learning techniques to build various models to solve problem that from real-life.

## Conclusion
>* The features“ram” and “battery_power” to be the most important variable for predicting the price range of mobile phone.
>* The MLP is the best model for predicting price range among all of the models that I built.
