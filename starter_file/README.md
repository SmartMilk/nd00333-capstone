*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Udacity Nanodegree - Machine Learning Engineer Capstone Project

For the final project of the Udacity Nanodegree - Machine Learning Engineer, I have chosen to use the popular UCI dataset Census Income dataset(referred to as the "salary" dataset). The main objective of this project is to construct models via AutoML and HyperDrive Azure models and select the best model that can predict whether an individual, based on their census characteristics, will have an annual salary exceeding $50k. The resultant best model is then deployed and validated as a REST endpoint on Azure Container Instance. 


## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

To download the dataset from Kaggle, a Kaggle account is required to create the authorization credentials. Once an account has been created, you can generate the json file containing the relevant credentials by going to Kaggle's account settings and selecting the 'generate api' option. Add the kaggle.json file to your working directory. No other special steps are required. 

![grep](https://github.com/SmartMilk/nd00333-capstone/blob/db32400d9a45d4483a6476339bf7cc3866f93633/starter_file/Proj_Images/kaggle_api_token_generate.jpg)

## Dataset

### Overview
*TODO*: Explain about the data you are using and where you got it from.

As outlined in the introduction, the dataset used in this project is the Census Income dataset, which can be found in the UCI repository. For this project, a copy of the dataset was extracted from a [Kaggle](https://www.kaggle.com/datasets/ayessa/salary-prediction-classification) dataset named "Salary Prediction Classification". The dataset consists of 14 census statistics from a 1994 census database with approximately 300,000 entries. 

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

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
*TODO*: Explain how you are accessing the data in your workspace.

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
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
Link to video: https://youtu.be/--lG2rcdYYY
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
