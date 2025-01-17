# Projet de Détection de Spam
Ce projet est un système de détection de spam qui utilise un modèle d'apprentissage automatique pour classer les messages comme spam ou non-spam. Le projet inclut une application web FastAPI pour servir le modèle et un script pour générer un grand jeu de données de messages spam et non-spam.  

## Prérequis
Python 3.8+
pip

## Installation
Clonez le dépôt :  
```bash
  git clone https://github.com/CCI-CDA/Spameur_Dassise.git
  cd Spameur_Dassise
```

## creation de l'envoironement virtuel
```
python3 -m venv venv
```

## activation de l'envirinement virtuel

```
source venv/bin/activate

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
  python train_model.py
```

## Lancement de l'Application Web

Pour lancer l'application web, exécutez le script suivant :
```bash
  python app.py
```

## Build de l'image Docker

```Pour mac ->
docker build -t grpccicdaacr.azurecr.io/spamer_dass .

docker build -t grpccicdaacr.azurecr.io/spamer_dassamd --platform=linux/amd64 .
```


## RUN
```
docker run -p 3333:3333 grpccicdaacr.azurecr.io/ameliespam
```
## Push
```
docker push grpccicdaacr.azurecr.io/VOTREIMAGE
docker push grpccicdaacr.azurecr.io/spamer_dassamd
```

## Docker pull

````
docker pull grpccicdaacr.azurecr.io/spamer_dassamd
````

## Lancer la VM
```
ssh ccicdauser@grpccicda.francecentral.cloudapp.azure.com
```
## pwd
```
CCIsurCDA2024
```

## url 

```
http://grpccicda.francecentral.cloudapp.azure.com:5600
```