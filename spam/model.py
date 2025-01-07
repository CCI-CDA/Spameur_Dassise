from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pandas as pd


def train_model():
    # Charger les données
    df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

    # Transformation des labels en numérique : spam = 1, ham = 0
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    # Séparation des données
    X = df['message']
    y = df['label']

    # Transformation des messages en vecteurs
    vectorizer = CountVectorizer()
    X_vect = vectorizer.fit_transform(X)

    # Entraînement du modèle Naive Bayes
    model = MultinomialNB()
    model.fit(X_vect, y)

    # Sauvegarder le modèle et le vectoriseur
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    print("Modèle et vectoriseur sauvegardés avec succès.")

def load_model():
    # Charger le modèle et le vectoriseur
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

def predict(text):
    model, vectorizer = load_model()  # Charger le modèle et le vectoriseur
    text_vect = vectorizer.transform([text])  # Appliquer la transformation avec le même vectoriseur
    prediction = model.predict(text_vect)  # Faire la prédiction
    return "Spam" if prediction[0] == 1 else "Ham"

if __name__ == '__main__':
    # Entraîner le modèle et sauvegarder
    train_model()
