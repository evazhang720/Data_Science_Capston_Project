
# coding: utf-8

# ## Introduction
# >* This is data capstone project. The dataset was taken from Kaggle. The full database contains 21 features of mobile phone of varies companies.There are two datasets, one is train.csv(2000 rows and 21 columns), another is test.csv (1000 rows and 21 columns). The aim of this project is “find out the relation between features of mobile phone and using the machine learning technique to predict the right price range of a mobile phone in the competitive mobile phone market”.

# ## 1. Load Packages and Datasets

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import missingno as msno
import math
import random
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score,f1_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle
from sklearn import svm, datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data as data_utils

# import data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


# ## 2. Data Overview

# check the data frame for train dataset
df_train.head()

## check the data frame for test dataset
df_test.head()

all_columns_train = df_train.columns.tolist()
all_columns_train

all_columns_test = df_test.columns.tolist()
all_columns_test


# #### Data Description
# >* This dataset contains 42000 observations, and 21 variables.
# 
# | Variable Name: | Variable Description: | 
# | -- | -- | 
# |battery_power|Total energy a battery can store in one time measured in mAh   
# |blue|Has bluetooth or not   
# |clock_speed|speed at which microprocessor executes instructions  
# |dual_sim|Has dual sim support or not   
# |fc|Front Camera mega pixels  
# |four_g|Has 4G or not        
# |int_memory|Internal Memory in Gigabytes  
# |m_dep|Mobile Depth in cm  
# |mobile_wt|Weight of mobile phone  
# |n_cores|Number of cores of processor   
# |pc|Primary Camera mega pixels              
# |px_height|Pixel Resolution Height   
# |px_width|Pixel Resolution Width   
# |ram|Random Access Memory in Mega Bytes  
# |sc_h|Screen Height of mobile in cm   
# |sc_w|Screen Width of mobile in cm  
# |talk_time|longest time that a single battery charge will last when you are   
# |three_gHas|3G or not  
# |touch_screen|Has touch screen or not  
# |wifi|Has wifi or not   
# |price_range|This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost) 

# get dimensionally of our dataset
train_shape=df_train.shape
test_shape=df_test.shape
print(train_shape)
print(test_shape)

df_train.dtypes
df_test.dtypes


# ## 3. Initial Statistics Analysis for Training set

df_train.describe().T


# ## 4. Exploratory data analysis
# ##### 1）Check Missing Value

# find the number of missing values for each variable of train dataset
msno.matrix(df_train.sample(2000),
            figsize=(16, 7),
            width_ratios=(15, 1))
df_train.isnull().sum() 

# find the number of missing values for each variable of test dataset
msno.matrix(df_test.sample(1000),
            figsize=(16, 7),
            width_ratios=(15, 1))
df_test.isnull().sum()


# ##### 2) Handle with Categorial Variable

# find out the number of columns that are "object" in training set
print("Data types and their frequency\n{}".format(df_train.dtypes.value_counts()))

#get format for value of each object column for training set
object_columns_df = df_train.select_dtypes(include=['object'])
print(object_columns_df.iloc[0])

#explore the number of unique value of each object column in traing set
cols = ['blue','wifi']
for name in cols:
    print(name,':')
    print(object_columns_df[name].value_counts(),'\n')
#every obejct column in my dataset contain discrete categorical values
#I can use the one-hot encoding to handle them

# Since the value of our obeject columns are the nominal values,
#I encode them as dummy variables

dummy_df_train = pd.get_dummies(df_train[cols])
dummy_df_train.head()

data_train = pd.concat([dummy_df_train,df_train], axis=1)
data_train = data_train.drop(cols, axis=1)
data_train.head()

# find out the number of columns that are "object" in test set
df_test.head()
print("Data types and their frequency\n{}".format(df_test.dtypes.value_counts()))

#get format for value of each object column for test set
object_columns_df = df_test.select_dtypes(include=['object'])
print(object_columns_df.iloc[0])

#explore the number of unique value of each object column in test set
cols = ['blue','wifi']
for name in cols:
    print(name,':')
    print(object_columns_df[name].value_counts(),'\n')
#every obejct column in my dataset contain discrete categorical values
#I can use the one-hot encoding to handle them

dummy_df_test = pd.get_dummies(df_test[cols])
data_test = pd.concat([df_test, dummy_df_test], axis=1)
data_test = data_test.drop(cols, axis=1)
data_test.head()


# ####  3) Narrow Down Columns
# ##### <1> Delete Columns

# Train the Random Forest model for raw data
# to check whether there have the useless columns or not
train_x = data_train.iloc[:, :22].values
train_y = data_train.iloc[:, 22].values
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0, random_state=0)
random_forest_model = RandomForestClassifier(n_estimators=10,max_features=int(math.sqrt(22)), max_depth=None,min_samples_split=2, bootstrap=True)
random_forest_model.fit(x_train, y_train)

# Get the importance of each feature to price range 
importance_list = random_forest_model.feature_importances_
feature_list = data_train.columns.values.tolist()
feature_dict = {}
for i in range(len(importance_list)):
    feature_dict[feature_list[i]] = [importance_list[i]]
feature_frame = pd.DataFrame(feature_dict)
feature_frame = feature_frame.T.sort_values([0], ascending=False).T
feature_frame.head()
#from output, we can clearly see there are no useless columns
# "ram" and "battery_power" are most important to the price range

# calculate the correlation of each variable
# to see whether there have linearly dependent columns or not
corr = data_train.corr()
plt.figure(figsize=(16,17))
sb.heatmap(corr, annot=True)
plt.show()
# Here, I found the indicators "pc" and "fc" have the strong correlation (absolute value is greater than 0.6)
# I decided to drop the "fc"

drops=['fc']
data_train = data_train.drop(drops, axis=1)
data_train.head()

drops=['fc']
data_test = data_test.drop(drops, axis=1)
data_test.head()


# In[378]:


# Encode three_g and four_g into network
def encodeNetwork(three_g, four_g):
    if three_g:
        if four_g:
            return 4
        else:
            return 3
    else:
        if four_g:
            return 1
        else:
            return 0

col_name = data_train.columns.tolist()
col_name.insert(1, 'network')
data_train['network'] = data_train.apply(lambda record: encodeNetwork(record.three_g, record.four_g), axis = 1)
data_train = data_train.reindex(columns=col_name)
data_train.drop(labels=['three_g', 'four_g'], axis=1, inplace= True)
data_train.head()
#“3” indicates “three_g”, “3” indicates “four_g”, “0” indicates neither “three_g” nor “four_g”. 

data_train.shape

data_test['network'] = data_test.apply(lambda record: encodeNetwork(record.three_g, record.four_g), axis = 1)
data_test.drop(labels=['three_g', 'four_g'], axis=1, inplace= True)
data_test.head()

data_test.shape

data_train.shape

#make sure all indicators have same length and see the shape of our data set
data_train.info()
print(data_train.shape)

data_test.info()
print(data_test.shape)


# #### 4) Visualization of  Remaining Variable!
# #####  Univariate Analysis
# ##### Independent Variable

# visualizate the remaining variable
var_list = data_train.columns.values.tolist()[:20]
for label in var_list:
    data_train[label].hist(edgecolor='white', grid=True)
    plt.xlabel(label)
    plt.ylabel("Number")
    plt.title("Histogram of "+label)
    plt.show()


# ##### Target Value Analysis-dependent variable

price_types=pd.value_counts(data_train['price_range'], sort= True).sort_index()
price_types.plot(kind = 'bar')
plt.title('Types of Price Range')
plt.xlabel('Price_range')
plt.ylabel("Number")
price_types=pd.value_counts(data_train['price_range'], sort= True).sort_index()
plt.show();
#each price range has the same number
#which means the data is balance

price_types.plot(kind = 'pie')
plt.title('Types of Price Range')
plt.show

corr = data_train.corr()
plt.figure(figsize=(16,17))
sb.heatmap(corr, annot=True)
plt.show()


# #### Firstly, we see there are moderate correlation(>0.5) between  "px_width" and "px_height", "sc_w" and "sc_h". Therefore, we want to further explore the relationship between these features.

# ##### 1)The relationship between  'px_width' and 'px_height'

#use the jointplot to see the binary distribution
sb.jointplot(x='px_width',y='px_height',kind="kde", color="r",data=data_train)


# ##### 3) The relationship between "sc_w" and "sc_h"

sb.jointplot(x='sc_w',y='sc_h',kind="kde", color="g",data=data_train)


# #### Secondly, from the heatmap, we can also clearly find the "ram", "battery_power" are most related to our target varible- "price_range". Therefore, I will explore the relationship between these features and price-range.

# ##### How does price-range affect by the ram

sb.violinplot(x='price_range',y='ram', data=data_train)
#the wide space represents more data in this interval 
#for the lower price (price range=0), the ram of the most mobile phone is also small; 
#the higher price (price range = 3), the ram of the most phone is also large.


# #### Boxplot
# >* N=1.5IQR  
# >* Q1-N > Outliers > Q3+N

sb.boxplot(x="price_range", y="ram", data=data_train)
# price range “0”, “2”, “3” have outliers


# ##### How does price_range affect by the battery_power

sb.violinplot(x='price_range',y='battery_power',data=data_train)
#the positive correlation is not particularly obvious, especially for the price interval “1” and “2”
#for the price “0”, most samples have small battery capacity
#but there are most of samples have big battery capacity in the price range “3”.


sb.boxplot(x="price_range", y="battery_power", data=data_train)

# overall relationship between the “ ram”, “battery_power”, and “price_range” to make the project diversity
sb.swarmplot(x='price_range',y='ram',data=data_train,color='orange')
sb.swarmplot(x='price_range',y='battery_power',data=data_train,color='lightblue')


# ## 3. Model-Machine Learning Part

# #### Data Pre-processing

# Prepare data for model
def splitTrainTestSetBin(dataframe, per):
    df_x = dataframe.iloc[:, :20]
    df_x_norm = (df_x - df_x.min()) / (df_x.max() - df_x.min())
    x = df_x_norm.values
    y = dataframe.iloc[:, 20].values
    # Binarize the output
    y_binary = label_binarize(y, classes=[0, 1, 2, 3])
    n_classes = y_binary.shape[1]
    #n_classes = 4
    x_train, x_test, y_train, y_test = train_test_split(x, y_binary, test_size=per, random_state=0)
    return x_train, x_test, y_train, y_test, n_classes

def splitTrainTestSet(dataframe, per):
    df_x = dataframe.iloc[:, :20]
    df_x_norm = (df_x - df_x.min()) / (df_x.max() - df_x.min())
    x = df_x_norm.values
    y = dataframe.iloc[:, 20].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=per, random_state=0)
    return x_train, x_test, y_train, y_test

# Get the training/testing dataset
train_x_binary, test_x_binary, train_y_binary, test_y_binary, num_classes = splitTrainTestSetBin(data_train, 0.3)
train_x, test_x, train_y, test_y = splitTrainTestSet(data_train, 0.3)
# Convert the data into the same scale
stdsc = StandardScaler()
train_x = stdsc.fit_transform(train_x)
test_x = stdsc.transform(test_x)


# #### Machine Learning Part
# ##### a) Naive Bayes

# Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(train_x, train_y)

#test the model
pred_y = nb_clf.predict(test_x)

#Calculate accuracy to compare the prediction with the target value
accuracy_nb_clf = []
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
for i in range(len(test_y)):
    label = test_y[i]
    class_correct[label] += (test_y[i]==pred_y[i])
    class_total[label] += 1
    
for i in range(num_classes):
    accuracy_nb_clf.append(np.round(100 * class_correct[i] / class_total[i], 2))
    

# Calculate the precision score
precision_nb_clf = np.round(precision_score(test_y, pred_y, average=None)*100,2)

# Calculate recall score
recall_nb_clf = np.round(recall_score(test_y, pred_y, average=None)*100,2)

#Calculate the F1 score
f1_nb_clf = np.round(f1_score(test_y, pred_y, average=None)*100,2)
print('The accuracy of naive bayes classifier is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (accuracy_nb_clf[0], accuracy_nb_clf[1], accuracy_nb_clf[2], accuracy_nb_clf[3]))
print('The precision of naive bayes classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (precision_nb_clf[0], precision_nb_clf[1], precision_nb_clf[2], precision_nb_clf[3]))
print('The recall score of naive bayes classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (recall_nb_clf[0], recall_nb_clf[1], recall_nb_clf[2], recall_nb_clf[3]))
print('The f1 score of naive bayes classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (f1_nb_clf[0], f1_nb_clf[1], f1_nb_clf[2], f1_nb_clf[3]))

# Naive Bayes
nb_clf = OneVsRestClassifier(GaussianNB())
nb_clf.fit(train_x, train_y_binary)

#test the model
pred_y = nb_clf.predict(test_x_binary)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y_binary[:, i], pred_y[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_y_binary.ravel(), pred_y.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
lw = 2
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.show()


# #### In conclusion, the Naïve Bayes Classifier works well for the price range 1

# ##### b) Support Vector Machine (SVM)

# SVM
svm_clf = svm.LinearSVC(C=5.0,max_iter=10000)
svm_clf.fit(train_x, train_y)
pred_y = svm_clf.predict(test_x)

#Calculate accuracy
accuracy_svm_clf = []
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
for i in range(len(test_y)):
    label = test_y[i]
    class_correct[label] += (test_y[i]==pred_y[i])
    class_total[label] += 1
    
for i in range(num_classes):
    accuracy_svm_clf.append(np.round(100 * class_correct[i] / class_total[i], 2))

# Calculate the precision score
precision_svm_clf = np.round(precision_score(test_y, pred_y, average=None)*100,2)

# Calculate recall score
recall_svm_clf = np.round(recall_score(test_y, pred_y, average=None)*100,2)

#Calculate the F1 score
f1_svm_clf = np.round(f1_score(test_y, pred_y, average=None)*100,2)
print('The accuracy of svm classifier is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (accuracy_svm_clf[0], accuracy_svm_clf[1], accuracy_svm_clf[2], accuracy_svm_clf[3]))
print('The precision of svm classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (precision_svm_clf[0], precision_svm_clf[1], precision_svm_clf[2], precision_svm_clf[3]))
print('The recall score of svm classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (recall_svm_clf[0], recall_svm_clf[1], recall_svm_clf[2], recall_svm_clf[3]))
print('The f1 score of svm classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (f1_svm_clf[0], f1_svm_clf[1], f1_svm_clf[2], f1_svm_clf[3]))

svm_clf = OneVsRestClassifier(svm.LinearSVC(C=5.0,max_iter=10000))
svm_clf.fit(train_x_binary, train_y_binary)
pred_y = svm_clf.predict(test_x_binary)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y_binary[:, i], pred_y[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_y_binary.ravel(), pred_y.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
lw = 2
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Support Vector Machine')
plt.legend(loc="lower right")
plt.show()


# #### Thus, SVM classifier has the better performance on price range 0 and price range 3.

# #### 1) Decision Tree

# Decision Tree
bdt_discrete = DecisionTreeClassifier(max_depth=2)
bdt_discrete.fit(train_x, train_y)
pred_y = bdt_discrete.predict(test_x)

#Calculate accuracy
accuracy_dt_clf = []
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
for i in range(len(test_y)):
    label = test_y[i]
    class_correct[label] += (test_y[i]==pred_y[i])
    class_total[label] += 1
    
for i in range(num_classes):
    accuracy_dt_clf.append(np.round(100 * class_correct[i] / class_total[i], 2))


# Calculate the precision score
precision_dt_clf = np.round(precision_score(test_y, pred_y, average=None)*100,2)

# Calculate recall score
recall_dt_clf = np.round(recall_score(test_y, pred_y, average=None)*100,2)

#Calculate the F1 score
f1_dt_clf = np.round(f1_score(test_y, pred_y, average=None)*100,2)

print('The accuracy of decision classifier is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (accuracy_dt_clf[0], accuracy_dt_clf[1], accuracy_dt_clf[2], accuracy_dt_clf[3]))
print('The precision of decision tree classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (precision_dt_clf[0], precision_dt_clf[1], precision_dt_clf[2], precision_dt_clf[3]))
print('The recall score of decision tree classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (recall_dt_clf[0], recall_dt_clf[1], recall_dt_clf[2], recall_dt_clf[3]))
print('The f1 score of decision tree classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (f1_dt_clf[0], f1_dt_clf[1], f1_dt_clf[2], f1_dt_clf[3]))

bdt_discrete = OneVsRestClassifier(DecisionTreeClassifier(max_depth=2))
bdt_discrete.fit(train_x_binary, train_y_binary)
pred_y = bdt_discrete.predict(test_x_binary)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y_binary[:, i], pred_y[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_y_binary.ravel(), pred_y.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= num_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
lw = 2
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Naive Bayes Classifier')
plt.legend(loc="lower right")
plt.show()


# #### The results show the Decision Tree classifier has the better performance on price range 0 and price range 3 compare to other price range

# ### Neural Network Part
# ##### Multilayer Perceptron

# Machine Learning - MLP
BATCH_SIZE = 50
NUM_EPOCHS = 2000
LEARNING_RATE = 0.005
HIDDEN_NODE = 100
SEED_START = 10
LOSS_TARGET = 0.01
SEED_STEP = 5
# Model defined in PyTorch
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.inputLayer = torch.nn.Linear(20, HIDDEN_NODE)
        self.hiddenLayer1 = torch.nn.Linear(HIDDEN_NODE, HIDDEN_NODE)
        self.hiddenLayer2 = torch.nn.Linear(HIDDEN_NODE, HIDDEN_NODE)
        self.hiddenLayer3 = torch.nn.Linear(HIDDEN_NODE, HIDDEN_NODE)
        self.hiddenLayer4 = torch.nn.Linear(HIDDEN_NODE, HIDDEN_NODE)
        self.hiddenLayer5 = torch.nn.Linear(HIDDEN_NODE, HIDDEN_NODE)
        self.outputLayer = torch.nn.Linear(HIDDEN_NODE,4)
        
        
    def forward(self, inputs):
        tmpOut = torch.nn.functional.relu(self.inputLayer(inputs))
        tmpOut = torch.nn.functional.relu(self.hiddenLayer1(tmpOut))
        tmpOut = torch.nn.functional.relu(self.hiddenLayer2(tmpOut))
        tmpOut = torch.nn.functional.relu(self.hiddenLayer3(tmpOut))
        tmpOut = torch.nn.functional.relu(self.hiddenLayer4(tmpOut))
        tmpOut = torch.nn.functional.relu(self.hiddenLayer5(tmpOut))
        resOut = torch.nn.functional.relu(self.outputLayer(tmpOut))
        return resOut

best_loss = 1

# Prepare data loader for PyTorch
train = data_utils.TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
train_loader = data_utils.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
test = data_utils.TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))
test_loader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

mlp_model = MLP()
best_model = mlp_model

while best_loss>LOSS_TARGET:
    random.seed(SEED_START)
    SEED_START += SEED_STEP
    mlp_model = MLP()
    
    # cost func and optim
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=LEARNING_RATE, momentum=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(NUM_EPOCHS):
        for i, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()

            x, y = Variable(data), Variable(target)
            #print y.squeeze()

            outputs = mlp_model(x)
            #print outputs
            #print y
            #print outputs
            loss = criterion(outputs, y)
            # Run back-propagation
            loss.backward()

            optimizer.step()

            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch+1, NUM_EPOCHS, i+1, len(train)//BATCH_SIZE, loss.data[0]))
            if loss.data[0]<best_loss:
                best_model = mlp_model
                best_loss = loss.data[0]
                print('Find a better model with loss: %.4f' % best_loss)

class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))
test_y_mlp = []
predict_y_mlp = []
accuracy_mlp = []
with torch.no_grad():
    for data in test_loader:
        x, y = data
        x, y = Variable(x), Variable(y)
        outputs = mlp_model(x)
        _, predicted = torch.max(outputs, 1)
        test_y_mlp.extend(y.numpy())
        predict_y_mlp.extend(predicted.numpy())
        c = (predicted == y).squeeze()
        for i in range(BATCH_SIZE):
            label = y[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(num_classes):
    accuracy_mlp.append(np.round(100 * class_correct[i] / class_total[i], 2))
    print('Accuracy of Class %d : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))

# Calculate the precision score
precision_mlp = np.round(precision_score(test_y_mlp, predict_y_mlp, average=None)*100,2)

# Calculate recall score
recall_mlp = np.round(recall_score(test_y_mlp, predict_y_mlp, average=None)*100,2)

#Calculate the F1 score
f1_mlp = np.round(f1_score(test_y_mlp, predict_y_mlp, average=None)*100,2)
print('The precision of mlp classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (precision_mlp[0], precision_mlp[1], precision_mlp[2], precision_mlp[3]))
print('The recall score of mlp classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (recall_mlp[0], recall_mlp[1], recall_mlp[2], recall_mlp[3]))
print('The f1 score of mlp classifier on test set is %2.2f(Class 0), %2.2f(Class 1), %2.2f(Class 2), %2.2f(Class 3).'% (f1_mlp[0], f1_mlp[1], f1_mlp[2], f1_mlp[3]))


# ##### All index of all price range bigger than 80% which is good

# # 4. Conclusion

# ### Models Comparison

# Compare the performance of different model
compareDict = {
    'Accuracy': {
        'Naive Bayes': np.mean(accuracy_nb_clf),
        'SVM': np.mean(accuracy_svm_clf),
        'Decision Tree': np.mean(accuracy_dt_clf),
        'MLP': np.mean(accuracy_mlp)
    },
    'Precision': {
        'Naive Bayes': np.mean(precision_nb_clf),
        'SVM': np.mean(precision_svm_clf),
        'Decision Tree': np.mean(precision_dt_clf),
        'MLP': np.mean(precision_mlp)
    },
    'Recall Score': {
        'Naive Bayes': np.mean(recall_nb_clf),
        'SVM': np.mean(recall_svm_clf),
        'Decision Tree': np.mean(recall_dt_clf),
        'MLP': np.mean(recall_mlp)
    },
    'F1 Score': {
        'Naive Bayes': np.mean(f1_nb_clf),
        'SVM': np.mean(f1_svm_clf),
        'Decision Tree': np.mean(f1_dt_clf),
        'MLP': np.mean(f1_mlp)
    }
}

print(compareDict)

compareDF = pd.DataFrame(compareDict)

compareDF.head()


# ##### It clearly show that the performance of MLP is better than other models and every indictor for MLP is bigger than 90%
# ##### Therefore, I decided to choose the MLP as my model to predict the test.csv

# #### Apply MLP to the test.csv

# Choose MLP as best model and apply to test file
df_x = data_test.iloc[:, :20]
df_x_norm = (df_x - df_x.min()) / (df_x.max() - df_x.min())
test_x = df_x_norm.values
test_y = list(0. for i in range(len(test_x)))
test = data_utils.TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))
test_loader = data_utils.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
predict_y = []
with torch.no_grad():
    for data in test_loader:
        x, y = data
        x, y = Variable(x), Variable(y)
        outputs = mlp_model(x)
        _, predicted = torch.max(outputs, 1)
        predict_y_mlp.extend(predicted.numpy())

# Display the predict result
print (predict_y_mlp)

# #### In conclusion, there are store correlation between "fc" and "pc”. Besides, the ‘ram’, 'battery_power' is the most important variable for the price range, and MLP is best model to predict price range among all of the models that I built.
