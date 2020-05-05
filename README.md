# Create a Customer Segmentation Report for Arvato Financial Solutions

[//]: # (Image References)

[image1]: ./images/missing_value.png "missing_value"
[image2]: ./images/formatit_missing_value.png "formatit_missing_value"
[image3]: ./images/categorical.png "categorical"
[image4]: ./images/quantitative.png "quantitative"
[image5]: ./images/cap_outlier.png "cap_outlier"
[image6]: ./images/drop.png "drop"
[image7]: ./images/pca.png "pca"
[image8]: ./images/kmeans.png "kmeans"
[image9]: ./images/cluster.png "cluster"
[image10]: ./images/diff_feature.png "diff_feature"
[image11]: ./images/train.png "train"
[image12]: ./images/model.png "model"
[image13]: ./images/xgboost.png "xgboost"




### Table of Contents

1. [Introduction](#introduction)
1. [Installation](#installation)
1. [File Descriptions](#files)
1. [Results](#results)
1. [Terms & Conditions](#terms)


### Introduction <a name="introduction"></a>:
The project has three major steps: the customer segmentation report, the supervised learning model, and the Kaggle Competition.

1. Customer Segmentation Report
This section will be similar to the corresponding project in Term 1 of the program, but the datasets now include more features that you can potentially use. You'll begin the project by using unsupervised learning methods to analyze attributes of established customers and the general population in order to create customer segments.

2. Supervised Learning Model
You'll have access to a third dataset with attributes from targets of a mail order campaign. You'll use the previous analysis to build a machine learning model that predicts whether or not each individual will respond to the campaign.

3. Kaggle Competition
Once you've chosen a model, you'll use it to make predictions on the campaign data as part of a Kaggle Competition. You'll rank the individuals by how likely they are to convert to being a customer, and see how your modeling skills measure up against your fellow students.


### Installation: <a name="installation"></a>:
Below are some libraries we used.
* Python 3.6.8
* numpy 1.16.4
* pandas 0.24.2
* matplotlib 3.0.2
* seaborn 0.9.0
* scikit-learn 0.21.2
* xgboost 0.90
* ipython 6.1.0
* ipython-genutils 0.2.0
* jupyter-client 5.1.0
* jupyter-console 5.2.0
* jupyter-core 4.3.0
* jupyterlab 0.35.6
* jupyterlab-launcher 0.4.0
* jupyterlab-server 0.2.0

Or you can run below command to setup the environment.
```
    conda create --name arvato-project python=3.6
	source activate arvato-project
	pip install -r requirements/requirements.txt
```


### File Descriptions: <a name="files"></a>:
```
- terms_and_conditions
|- terms.md                     : Terms and Condition
|- terms.pdf                    : Terms and Condition
- Arvato Project Workbook.ipynb : Main Project File in Notepad format
- Arvato Project Workbook.html  : Main Project File in HTML format
- requirements.txt              : Required Software and Plugins
- README.md                     : Project Description with Result and Discussion
```


### Results and Discussion: <a name="results"></a>:
This is a practical walk-through of customer clustering and prediction. If you are interested in machine learning. If you are interested in how machine learning can help identify target customers, you are the best audience of this article. I will give you a brief guide on how to use machine learning to solve clustering and prediction.
First, I will give you some brief introduction to the dataset. Below is some description of it. We will use the 4 spreadsheets below to help the company identify its target customers.
```
Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
```
And these 2 are the metadata to help describe above spreadsheets.
* DIAS Information Levels — Attributes 2017.xlsx: a top-level list of attributes and descriptions, organized by informational category
* DIAS Attributes — Values 2017.xlsx: detailed mapping of data values for each feature in alphabetical order.
This project is split into three parts. First, I explore the dataset and find the data characteristics. Then, use unsupervised learning to do customer clustering. Finally, I will use supervised learning to make a model to predict customer. So, don’t stop. Let’s get started on my journey.
#### Part 0: Get to Know the Data:
* Data Preprocessing and Explanation:
In this section, I am sequentially exploring the spreadsheets and summarize some data structure for further data preprocessing. This part is very important but really tedious. Because the further analyze all dependent on this part. In this part, I will separate the data into 5 types: binary, nominal, ordinal, mixed category, and quantitative feature. Because the data preprocessing step for these features have a little different. I will also check the different unknown value of these features. But first, let’s take a look at the missing value distribution of the general population and customers.

    * Missing value distribution 

    ![alt text][image1]

After fixing the typo or document error, Move unknown value to NA. We will get the distribution as below.

    * Missing value distribution after change unknown value to NA

    ![alt text][image2]

To prevent data missing too large, We will choose 70% as a missing value threshold to drop features. With this drop, We will lose around 5x features.

And we will start to drop features according to their type, With the # unique value box plot as below. We can find that some documented/undocumented nominal or ordinal features have a little feature have # unique value more than 15, We will set 15 as # unique value drop threshold to drop nominal or ordinal categorical feature.

    * Categorical feature

    ![alt text][image3]

For mixed type categorical feature, we will split or translate it into the nominal or ordinal feature.

And for quantitative feature…

    * Quantitative feature

    ![alt text][image4]

We can find many features are right-skewed. After a log transform, outlier caping, and some further analyze. We can get more normalized distribution as below.

    * Quantitative feature after log transform and outlier caping

    ![alt text][image5]

After we actually drop the features, we can get below missing value distribution.
    * Drop

    ![alt text][image6]

Then we will start to check population feature coverage with a customer, If it is outside the bounds of population, we will NA it. And, start to fill NA with meaning value. For categorical feature, we will fill max occurs value, For quantitative feature, we will fill median. After that, apply one-hot encoding for the nominal feature. And the end of this part, I create a clean_data function for easily data preprocessing.

#### * Part 1: Customer Segmentation Report
In this part, I will use the data created in Part 0 to do customer segmentation. First, I use [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to scale the data to a z-score space and apply [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

    * PCA

    ![alt text][image7]

We can find that 90% variance is bounded by the first 200 dimensions. Then, I use [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) to fast find optimized K for [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) on these 200 dimensions.

    * KMeans

    ![alt text][image8]

According to the elbow method, we set K = 20 for KMeans. And apply KMeans, we can get 20 clusters as below.

    * Cluster

    ![alt text][image9]

We can find that cluster 4 and 2 are the most customers like and unlike cluster. And we further analyze the feature difference between these 2 clusters.

    * Feature difference

    ![alt text][image10]

Finally, we have some conclusions about the customers. They drive the small car, younger than 25 years old, low-income owners, no household, traditional mind, no luxury car, don’t like online purchase.

#### * Part 2: Supervised Learning Model

In this part, I will pick a machine-learning algorithm to train and predict customer. First, let’s take a look at our training data.

    * Data distribution

    ![alt text][image11]

We can see that this is an unbalanced dataset. After data cleaning and scaling, I will use [StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html), because it can preserve original data percentage in each fold. For metrics selection, because finally, this project will submit to Kaggle. I use [AUC for the ROC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve), which is as same as Kaggle. And for model selection, We pick 4 models to compare their performance: [Decision Tree](https://scikit-learn.org/stable/modules/tree.html), [SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), [XGBoost](https://xgboost.readthedocs.io/en/latest/). And finally, use [XGBoost](https://xgboost.readthedocs.io/en/latest/) as it is outstanding compared to the others.

    * Model selection

    ![alt text][image12]

After fine-tuning the hyperparameters, I got this top 20 feature importance score.

    * XGBoost

    ![alt text][image13]


From above, we can find D19_SOZIALES, D19_KONSUMTYP_MAX is the most important factor to affect the result, but we have no document describe them. But, at lease from the remaining features, we can let the company pay more attention to their customer’s traditional mind, money save, household and luxury car owner.

The final training AUC score for this project is 0.814929 and Kaggle score is 0.79872, rank 16. I still find ways to improve my ranking. Now, I am rank 10. Hope I can find another way to further increase my rank. Although this project is really challenging but really interesting!

#### Reflection
With this project, I think I have a more deep understanding of what machine learning can do. From data cleaning to unsupervised machine learning, PCA, dimensionality reduction, KMeans clustering, data recovery. And supervised machine learning, model selection, hyperparameter tuning, and prediction. It’s really a remindful experience. Hope someday I can get a chance to use this technique in other fields, such as medicine, education, engineering, …. There are many things I can do!

#### Improvement
For this section, I think I have some portion to improve

1. Code modularization: I think I can create a class to let anyone easily cluster and train the data. Maybe I can create a transform() method to let anyone easily clean the data.
2. Data cleaning: I think there should be another good method to let the data more purify, such as how to remove the feature? How to decide if this feature can safely remove? How to fill NA?
3. Model selection: Is XGBoost the best one? Can we combine many models or use [VotingClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)?

#### Conclusion
This project is really not an easy job. But I did it! Just share my experience with you. See you, and enjoy it.


### Terms & Conditions <a name="terms"></a>:
In addition to Udacity's Terms of Use and other policies, your downloading and use of the AZ Direct GmbH data solely for use in the Unsupervised Learning and Bertelsmann Capstone projects are governed by the following additional terms and conditions. The big takeaways:

1. You agree to AZ Direct GmbH's General Terms provided below and that you only have the right to download and use the AZ Direct GmbH data solely to complete the data mining task which is part of the Unsupervised Learning and Bertelsmann Capstone projects for the Udacity Data Science Nanodegree program.

1. You are prohibited from using the AZ Direct GmbH data in any other context.

1. You are also required and hereby represent and warrant that you will delete any and all data you downloaded within 2 weeks after your completion of the Unsupervised Learning and Bertelsmann Capstone projects and the program.

1. If you do not agree to these additional terms, you will not be allowed to access the data for this project.
The full terms are provided in the workspace below. You will then be asked in the next workspace to agree to these terms before gaining access to the project, which you may also choose to download if you would like to read in full the terms.
