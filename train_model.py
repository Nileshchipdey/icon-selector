import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Load and prepare data
data = pd.read_csv('data.csv')
X = data['text']
y = data['icon']

# Create a pipeline that combines a vectorizer with a classifier
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X, y)

# Save the model
joblib.dump(pipeline, 'model.pkl')
