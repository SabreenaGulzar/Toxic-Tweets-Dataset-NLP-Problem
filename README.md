# Toxic-Tweets-Classification
## Objective:
The goal of this project is to build a machine learning model to classify tweets as either "Toxic" or "Non-Toxic" based on their content. This is a binary classification problem where the labels are:

Toxic (1)
Non-Toxic (0)

## Dataset:
Source: Kaggle - Toxic Tweets Dataset
Content: The dataset consists of a collection of tweets labeled as either toxic or non-toxic.

## Procedure:
### Data Preprocessing:

#### Load the Data: The dataset is loaded into a Pandas DataFrame.
#### Text Cleaning: The tweets are preprocessed by removing unwanted characters, stopwords, and applying other text normalization techniques.
#### Feature Extraction:
Bag of Words (BoW): A representation of text data where each word's occurrence in the tweet is counted.
TF-IDF (Term Frequency-Inverse Document Frequency): A more sophisticated method that considers the frequency of words in the document and across all documents to create a weighted word matrix.

## Modeling:
Several machine learning algorithms are applied to the features extracted from the tweets:
Decision Trees
Random Forest
Naive Bayes
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Each of these algorithms is trained on the training data to learn patterns associated with toxic and non-toxic tweets.

## Evaluation Metrics:
Precision: The ratio of true positives to the sum of true and false positives.
Recall: The ratio of true positives to the sum of true positives and false negatives.
F1-Score: The harmonic mean of precision and recall, giving a balanced measure.
Confusion Matrix: A table showing the true positives, true negatives, false positives, and false negatives.
ROC-AUC Curve: A plot showing the trade-off between True Positive Rate and False Positive Rate, with AUC representing the overall performance of the model.

## Best Performing Model:
Random Forest was selected as the best-performing model based on its overall performance across the evaluation metrics. It handles overfitting better than Decision Trees, provides feature importance, and works well with both small and large datasets.

## Deployment:
The trained Random Forest model is deployed using Streamlit, a web application framework for creating interactive web apps with Python.
User Interaction: The Streamlit app allows users to input a tweet and classify it as toxic or non-toxic using the trained model.
The app also displays the model's performance metrics and a confusion matrix, helping users understand the model's effectiveness.

##Streamlit App Summary:
Input: Users can enter a tweet into a text box.
Output: The app classifies the tweet as either toxic or non-toxic based on the model's prediction.
Model Metrics: Users can view the classification report, confusion matrix, and ROC-AUC curve to understand how well the model performs.
