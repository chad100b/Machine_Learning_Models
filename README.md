# Machine_Learning_Models

# Unit 11 - Risky Business
 
![Credit Risk](Images/credit-risk.jpg)

## Project Background

Mortgages, student and auto loans, and debt consolidation are just a few examples of credit and loans that people seek online. Peer-to-peer lending services such as Loans Canada and Mogo let investors loan people money without using a bank. However, because investors always want to mitigate risk, a client has asked that our firm assist them predict credit risk with machine learning techniques.

In order to assist our client, our firm will build and evaluate several machine learning models to predict credit risk using data one would typically see from peer-to-peer lending services. Credit risk is an inherently imbalanced classification problem (i.e. the number of good loans is much larger than the number of at-risk loans), so we had to employ different techniques for training and evaluating models with imbalanced classes. We primarily used the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

1. [Resampling](#Resampling)
2. [Ensemble Learning](#Ensemble-Learning)

- - -

## Files

[Resampling Starter Notebook](Starter_Code/credit_risk_resampling.ipynb)

[Ensemble Starter Notebook](Starter_Code/credit_risk_ensemble.ipynb)

[Lending Club Loans Data](Resources/LoanStats_2019Q1.csv.zip)

- - -

## Coding Summary

### Resampling

We used the [imbalanced learn](https://imbalanced-learn.readthedocs.io) library to resample the LendingClub data and build and evaluate logistic regression classifiers using the resampled data.

We completed the following tasks in order to determine our analysis:

1. Read the CSV into a DataFrame.

2. Split the data into Training and Testing sets.

3. Scale the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.

4. Use the provided code to run a Simple Logistic Regression:
    * Fit the `logistic regression classifier`.
    * Calculate the `balanced accuracy score`.
    * Display the `confusion matrix`.
    * Print the `imbalanced classification report`.

We then used over and under and combined sampling algorithms:

1. Oversample the data using the `Naive Random Oversampler` and `SMOTE` algorithms.

2. Undersample the data using the `Cluster Centroids` algorithm.

3. Over- and undersample using a combination `SMOTEENN` algorithm.


With each model we performed the following sequencing:

1. Train a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.

2. Calculate the `balanced accuracy score` from `sklearn.metrics`.

3. Display the `confusion matrix` from `sklearn.metrics`.

4. Print the `imbalanced classification report` from `imblearn.metrics`.


Our analysis answers the following questions from our client:

*Which model had the best balanced accuracy score?*
The SMOTE Oversampling model had the best balanced accuracy score of %99.52, taking the y_train training sample from 2,500 to 56,252.

*Which model had the best recall score?*
Both the SMOTE Oversampling and the SMOTEEN Combination (Over and Under) Sampling had the best recall score when analysing the Imbalanced Classification Report. However, when we perform the actual calculation of recall on the two models, TP/(TP + FN), we find that SMOTE betters SMOTEEN with a recall of 99.387% versus 99.371%, respectively.

*Which model had the best geometric mean score?*
Both the SMOTE Oversampling and the SMOTEEN Combination (Over and Under) Sampling had the best geodmetric mean score at 1.00.

### Ensemble Learning

For the Ensemble Models, we trained and compared two different ensemble classifiers to predict loan risk and evaluate each model. You used the following models: [Balanced Random Forest Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.BalancedRandomForestClassifier.html) and the [Easy Ensemble Classifier](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html). 

We completed the following tasks in order to determine our analysis:

1. Read the data into a DataFrame using the provided starter code.

2. Split the data into training and testing sets.

3. Scale the training and testing data using the `StandardScaler` from `sklearn.preprocessing`.


With each model we performed the following sequencing:

1. Train the model using the quarterly data from LendingClub provided in the `Resource` folder.

2. Calculate the balanced accuracy score from `sklearn.metrics`.

3. Display the confusion matrix from `sklearn.metrics`.

4. Generate a classification report using the `imbalanced_classification_report` from imbalanced learn.

5. For the balanced random forest classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score.


Our analysis answers the following questions from our client:
*Which model had the best balanced accuracy score?*
The Balanced Random Forest Classifier had the best balanced accuracy score at %72.00.

*Which model had the best recall score?*
The Balanced Random Forest Classifier had the best recall score with high risk at %59, low risk at %85, and a total average risk of %85.

*Which model had the best geometric mean score?*
The Balanced Random Forest Classifier had the best geiometric mean score of %71 versus the Easy Ensamble Classifier of %68.

*What are the top three features?*
The top three features of the Balanced Random Forest Classifier are "total_rec_prncp", "last_pymnt_amnt", and "total_pymnt".
