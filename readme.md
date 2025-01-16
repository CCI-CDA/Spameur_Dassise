# Projet de Détection de Spam
Ce projet est un système de détection de spam qui utilise un modèle d'apprentissage automatique pour classer les messages comme spam ou non-spam. Le projet inclut une application web FastAPI pour servir le modèle et un script pour générer un grand jeu de données de messages spam et non-spam.  

## Prérequis
Python 3.8+
pip

## Installation
Clonez le dépôt :  
```bash
  git clone https://github.com/ItsCastoor/spam-detection-thomas.git
  cd spam-detection-thomas
```

## Installez les paquets requis :  
```bash
  pip install -r requirements.txt
```
## Génération du Jeu de Données
Pour générer un jeu de données de 1 million de messages (500,000 spam et 500,000 non-spam), 
exécutez le script suivant :
```bash
  python remplissage.py
```

## Entraînement du Modèle

Pour entraîner le modèle, exécutez le script suivant :
```bash
  python train.py
```

## Lancement de l'Application Web

Pour lancer l'application web, exécutez le script suivant :
```bash
  python app.py
```