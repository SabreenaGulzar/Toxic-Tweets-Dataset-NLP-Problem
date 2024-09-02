import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('FinalBalancedDataset.csv')
    return df

# Load and prepare the data
def prepare_data(df):
    X = df['tweet']
    y = df['Toxicity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    st.write(X_train_tfidf)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer

# Train the model
def train_model(X_train_tfidf, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)
    return model

# Save the model and vectorizer
def save_model(model, vectorizer):
    joblib.dump(model, 'random_forest_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Load the model and vectorizer
def load_model_and_vectorizer():
    model = joblib.load('random_forest_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# Define the Streamlit app
def main():
    st.title("Toxic Tweets Classification")
    
    # Load data
    df = load_data()
    X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer = prepare_data(df)
    
    # Train and save the model
    if not (st.file_uploader("Upload a model", type="pkl")):
        model = train_model(X_train_tfidf, y_train)
        save_model(model, tfidf_vectorizer)
    else:
        model, tfidf_vectorizer = load_model_and_vectorizer()
    
    # User input
    tweet = st.text_area("Enter a tweet:")
    if st.button("Classify"):
        if tweet:
            tweet_tfidf = tfidf_vectorizer.transform([tweet])
            prediction = model.predict(tweet_tfidf)
            if prediction[0] == 1:
                st.write("The tweet is classified as Toxic.")
            else:
                st.write("The tweet is classified as Non-Toxic.")
        else:
            st.write("Please enter a tweet to classify.")
    
    # Show evaluation metrics
    if st.checkbox("Show evaluation metrics"):
        y_pred = model.predict(X_test_tfidf)
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        st.bar_chart(cm)

if __name__ == "__main__":
    main()
