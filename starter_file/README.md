# Udacity Nanodegree - Machine Learning Engineer Capstone Project

For the final project of the Udacity Nanodegree - Machine Learning Engineer, I have chosen to use the popular UCI dataset Census Income dataset(referred to as the "salary" dataset). The main objective of this project is to construct models via AutoML and HyperDrive Azure models and select the best model that can predict whether an individual, based on their census characteristics, will have an annual salary exceeding $50k. The resultant best model is then deployed and validated as a REST endpoint on Azure Container Instance. 


## Project Set Up and Installation

The project is conducted almost entirely on Microsoft Azure Machine Learning Studio. At minimum, you will need to have access priveleges equal or equivalent to the *'AzureML Data Scientist'* role as listed in Azure's [documentation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles?tabs=labeler) to be able to replicate the steps shown in this project. You may also require an Azure administrator to provide access to Azure's compute instances and compute clusters. 

To download the dataset from Kaggle, a Kaggle account is required to create the authorization credentials. Once an account has been created, you can generate the json file containing the relevant credentials by going to Kaggle's account settings and selecting the 'generate api' option. Add the kaggle.json file to your working directory. No other special steps are required. 

![kaggle page](https://github.com/SmartMilk/nd00333-capstone/blob/db32400d9a45d4483a6476339bf7cc3866f93633/starter_file/Proj_Images/kaggle_api_token_generate.jpg)

## Dataset

### Overview

As outlined in the introduction, the dataset used in this project is the Census Income dataset, which can be found in the UCI repository. For this project, a copy of the dataset was extracted from a [Kaggle](https://www.kaggle.com/datasets/ayessa/salary-prediction-classification) dataset named "Salary Prediction Classification". The dataset consists of 14 census statistics from a 1994 census database with approximately 300,000 entries. 

### Task

The machine learning task to be solved in this project is a prediction model that can determine whether an individual, based on their census statistics, will earn greater than a $50k annual salary. The census statistics, aka the features, are outlined below (as taken from the Kaggle dataset):

- **age**: continuous.
- **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- **fnlwgt**: continuous.
- **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- **education-num**: continuous.
- **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- **race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- **sex**: Female, Male.
- **capital-gain**: continuous.
- **capital-loss**: continuous.
- **hours-per-week**: continuous.
- **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
- **salary** (prediction variable): <=50K or >50K

### Access

The data is accessed from Kaggle APIs using authorization credentials. See "Project Set Up and Installation" for more details.

In its raw format, the data is unsuitable to be fed directly into the AutoML or HyperDrive modules, and must undergo a cleaning process. 

The dataset has been prior explored by Nikita Verma in a [medium](https://medium.com/analytics-vidhya/machine-learning-application-census-income-prediction-868227debf12) article. As exploratory data analysis is not the focus of the project/course, most of the key steps taken by the article are repeated in a concise format (with a few deviations) to prepare the dataset for AutoML/Hyperdrive (with random checks to ensure steps are consistent). In essence, the objective of the dataset cleaning is to consolidate categorical variables that either have similar logical meaning (e.g merging jobs in local and state government into a single title) or follow similar distributions in terms of their salary prediction. This helps to reduce the likeliness of outliers in niche categories from skewing the prediction model. 

The dataset is prepared in the following order:

1. For all categorical variables, replace missing values ('?') with the mode categorical value for that particular variable
2. Remove 'education_num' as a parameter as we will be using a different encoding system for education
3. For 'workclass' variable, merge 'Never-worked' into 'Without-pay' | merge 'State-gov' and 'Local-gov' into new descriptor 'Gov' | merge 'Self-emp-inc' into 'Private'
4. For 'education' variable, merge 'Prof-school' into 'Doctorate' descriptior | merge 'Assoc-acdm' and 'Assoc-voc' into new descriptor 'Assoc'
5. For 'marital-status' variable, merge 'Married-Civ-Spouse' and 'Married-AF-Spouse' into new descriptor 'Married-with-spouse' | merge 'Separated', 'Divorced', 'Widowed' and 'Married-spouse-absent' into new descriptor 'No-spouse'
6. For 'relationship' variable, merge 'Not-in-family', 'Own-child', 'Unmarried' and 'Other-relative' into new descriptor 'Other' (based on similar distributions for salary)
7. For 'race' variable, merge 'Amer-Indian-Eskimo', 'Other' into existing 'Others' descriptor
8. Label encode the categorical variables. While one hot is typically preferred for non-ordinal variables (of which all categorical variables in the dataset apply as), one hot encoding introduces around 90 additional parameters into the dataset. This hampers the ability to interpret the model in terms of featurization and can lead to exponential computational costs for HyperDrive and autoML.
9. Continuous variables with skewed distributions: 'fnlwgt', 'capital_gain' and 'capital_loss', treated with square-root transform, cube-root transform and cube-root transform respectively

This produces a cleaned dataset (salary_cleaned.csv) which is ready to be used in AutoML and Hyperdrive. To view the cleaning process in full, please refer to the automl.ipynb script. The HyperDrive script merely intakes the cleaned dataset produced from the automl script. 

## Automated ML

AutoML configuration is relatively simple. Aside from the target dataset (the cleaned dataset) and the target label, the only other parameters that were considered were the experiment timeout -set to 30 minutes to reduce on azure resource usage- and the primary metric for parameter tuning, which was chosen to be accuracy, a standard method of evaluating machine learning models. If this were a multiclass classification issue, or there was imperative to penalize models for false positives/false negatives, then alternative metrics such as F1 and weighted AUC would have been used. However the simplicity of the classification problem permits the applicability of accuracy. Finally, calculations were performed on a remote cluster with the maximum amount of concurrent iterations set to 5, allowing one iteration to be run on each provided node. 

The selection process of models by the AutoML algorithm is proprietary, however it can choose any one of the classification models listed from Azure documentation screenshot below. The model that provided the best performance is described in the results hereafter. 
![automl documentation](https://github.com/SmartMilk/nd00333-capstone/blob/master/starter_file/Proj_Images/AutoML_supported_models.jpg)

### Results

The completed AutoML experiment revealed that the best performing model was a complex Voting Ensemble classifier with a maximum mean recognition accuracy of 88.6%, as detailed in the screenshots below. The Voting Ensemble is comprised of 12 sub-classifiers; largely XGBoost and Random Forest classifiers each built out of their own set of weak learner models with their own weights. Given the inherent complexity of the AutoML model, it is difficult to comment on how the Voting Ensemble could have been further optimized, however the AutoML model could be improved simply by extending the experiment timeout variable to several hours or even days. This would increase the probability of the AutoML module finding a more optimized model solution. 

![rundetails automl](https://github.com/SmartMilk/nd00333-capstone/blob/master/starter_file/Proj_Images/autoML_rundetails_widget.jpg)

![automl bestrun model](https://github.com/SmartMilk/nd00333-capstone/blob/master/starter_file/Proj_Images/automl_fitted_model_properties_and_ID.jpg)

## Hyperparameter Tuning

For the HyperDrive approach, the Support Vector Classifier(SVC) was chosen as the base classifier due to its general ability to fit most models well, based on past experiences. The SVC has 3 parameters that are tuned in the HyperDrive module:

- **C**: The regularization parameter of the SVC; higher values create a reduction in variance but may lead to overfitting. After some initial trials, the optimal range of C was found to be on a uniform scale of 0.01 and 2. Exceeding a value of 2 significantly slowed down a run to take >10 minutes to complete. 

- **Kernel**: The type of kernel to be fitted to the SVC model, with a choice of radial basis function(rbf), linear, polynomial or sigmoid kernels. 

- **Polynomial**: In the case of a polynomial kernel, this parameter determines the polynomial's degree, and was limited to be between 1 and 8. The parameter is ignored otherwise. 

The parameter sampling method was chosen to be Bayesian optimization, hence there was no need to apply an early termination policy. The estimator was generated from a Python training script (*hp_train.py*). In order to ensure fair comparison with the AutoML model, the training process was set to optimize accuracy as the primary metric. Finally, the maximum number of HyperDrive runs was limited to 80, as per an auto-generated recommendation by Azure's CLI based on the number of input parameters. 


### Results

The completed HyperDrive run revealed that the SVC model could achieve up to maximum of 81.7% mean accuracy with C = 2, the kernel set to polynomial and with a degree of 7, the output of each run and the corresponding tuning parameters can be seen in the screenshots below. As the regulazation parameter is at its maximum value, this implies the model has a degree of underfitting, and could probably be improved by further increasing the maximum range of C. This of course means there will be greater computational cost. 

![rundetails hyperdrive](https://github.com/SmartMilk/nd00333-capstone/blob/master/starter_file/Proj_Images/hyperdrive_runs_and_performance.jpg)

![hyperdrive bestrun model](https://github.com/SmartMilk/nd00333-capstone/blob/master/starter_file/Proj_Images/best_hyperdrive_run_and_parameters.jpg)

## Model Deployment

As the AutoML model vastly outperformed the HyperDrive model, the AutoML model was further selected to be deployed as an endpoint. The model is deployed as a REST endpoint on an Azure container instance, using a simple configuration of 1 CPU core and 1GB memory limit to reduce Azure costs. The successfully deployed model is shown in the screenshot below. Application Insights was also enabled in order to monitor the health of the webservice, which can be seen in further detail in the project screencast. 

![healthy endpoint](https://github.com/SmartMilk/nd00333-capstone/blob/master/starter_file/Proj_Images/healthy_endpoint_active.jpg)

To validate the endpoint was working as intended, 2 test datapoints in a JSON-formatted parameter (matching the format of the cleaned salary dataset) are fed into the endpoint using the *requests* package. 2 corresponding results [1 0] were produced by the endpoint (1 indicating a greater than $50k salary, 0 less than), showing that the endpoint was working correctly. Evidence of the successful endpoint test are shown in the screenshot below. 

![endpoint test](https://github.com/SmartMilk/nd00333-capstone/blob/master/starter_file/Proj_Images/Deployment_test.jpg)

## Screen Recording
A screen recording showcasing the project can be accessed through this youtube link: https://youtu.be/--lG2rcdYYY


