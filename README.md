# Toxic-Tweets-Dataset-NLP-Problem
Problem statement: This dataset has a collection of Tweets. Its labelled as Toxic - 1, Non toxic - 0. Apply the NLP

methods to predict the toxicity of the tweets. 
Download the dataset from the following Kaggle Compitation https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset. All
the credits to the original collectors.

For an NLP problem like the Toxic Tweets dataset, the process typically involves the following steps:
## Step 1: Load the Toxic Tweets Dataset
First, load the dataset, which should have at least two columns: the text of the tweet and the label (indicating whether the tweet is toxic or not).

# Step 2: Preprocess the Text
Preprocessing involves cleaning the text by removing special characters, converting to lowercase, and removing stopwords.

# Step 3: Convert Text to TF-IDF
Next, convert the preprocessed tweets into a TF-IDF representation using TfidfVectorizer from scikit-learn.

# Step 4: Train a Classification Model
Let’s train a simple Logistic Regression model to classify the tweets as toxic or not.

# Step 5: Build a Simple Streamlit App for Deployment
Finally, you can deploy the model using Streamlit to allow users to input tweets and get predictions.
