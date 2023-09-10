# treue_technology_Startup_Prediction
## Startup Success Prediction using Machine Learning

![Startup Success](![images](https://github.com/imshubhamkore/treue_technology_Startup_Prediction/assets/128685230/e22fc82f-7f41-4d21-aed9-cb5ff3fed6d3)


## Overview

This GitHub repository contains a machine learning project aimed at predicting the success of startups. We leverage various data attributes and machine learning models to forecast the likelihood of a startup achieving success or facing challenges.

## Table of Contents
- Project Structure
- Data
- Machine Learning Models
- Evaluation

## Project Structure
In this project, machine learning is used to make predictions about startup company success based on factors like funding, location, industry, and team size. The goal is to help investors and entrepreneurs make more informed decisions about which startups to invest in or launch and whether the startup will be successful or not.

## Data
The dataset used in this project is sourced from Treue Technology. It contains information about several thousand startup companies, including their funding history, location, industry, team size, and other relevant features. The dataset requires cleaning and preprocessing to remove missing values and outliers and to convert categorical variables into numeric ones.

## Machine Learning Model
To predict the success of startups, we tested several machine learning models, including XGBoost Classifier (XBC), AdaBoost Classifier (ABC), Random Forest Classifier (RFC), and Gradient Boosting Classifier (GBC). For each model, we performed a grid search to find the optimal hyperparameters that maximize the accuracy of the model.

## Evaluation
We found that Random Forest Classifier (RFC), which we tested against all the other models, provided the best accuracy score of 0.85. A scatterplot of the feature weights of each model, which we also created, demonstrates that funding and industry are the two main indicators of a startup's success.


To run the code, you'll need to do the following steps :

- Import necessary libraries :
  pandas
  numpy
  seaborn
  scikit-learn
  plotly

- Read startup data.csv, a CSV file, and store the data in the dataset dataframe.

- Rename the "status" column to "is_acquired" and change the values of "acquired" and "operating" to "1" and "0," respectively.

- Produce a heatmap to determine whether various dataset features are correlated.

- By computing the interquartile range (IQR) and determining whether any value falls outside the range of [Q1 - 1.5IQR, Q3 + 1.5IQR], remove outliers from the dataset. Outlier values are those that fall outside of this range.

- Using the scikit-learn library's KNNImputer, fill in any missing values in the dataset's numerical features.

- Removing unused features and changing some features from object type to numeric type.

- Create a matrix of features' correlations with the target variable "is_acquired" and eliminate features with a correlation coefficient of less than 0.2.

- The main code functions that are used to carry out the aforementioned operations, such as "ignore_warn," "draw_heatmap," "getOutliersMatrix," and "imputing_numeric_missing_values."

- The cleaned and processed dataset is then saved in the dataframe named "dataset."

- Combine the predictions of various base models using the meta-modeling approach to increase the final prediction's accuracy. A Gradient Boosting Classifier (GBC), an AdaBoost Classifier (ABC), a Random Forest Classifier (RFC), and an XGBoost Classifier (XBC) serve as the foundational models.

- A grid search is conducted for each of the base models to identify the ideal hyperparameters that produce the highest accuracy score. The hyperparameters that are tuned vary depending on the model, but some common ones include the maximum depth of the tree, the number of estimators, and the learning rate.

- Print Every grid search yields a set of results, and the best estimator is added to a list of the best classifiers. Finally, define a function in Plotly to generate a scatterplot of the feature weights of each base model.




