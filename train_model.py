import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Charger les données
data = pd.read_csv('spam_data.csv', header=None, names=['message', 'label'], low_memory=False)

# Séparer les messages et les étiquettes
messages = data['message']
labels = data['label']

# Entraîner le vectorizer
vectorizer = CountVectorizer()
message_vectors = vectorizer.fit_transform(messages)

# Entraîner le modèle
model = MultinomialNB()
model.fit(message_vectors, labels)

# Sauvegarder le vectorizer
with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Sauvegarder le modèle
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)