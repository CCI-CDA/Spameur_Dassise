import logging
import pickle
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Body
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Optional
import json
from fastapi.middleware.cors import CORSMiddleware

# Initialisation de l'application FastAPI
app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Charger le modèle de détection de spams
try:
    with open('spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Charger le vectoriseur
try:
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info("Vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading vectorizer: {e}")
    raise

# Configuration de la base de données
DATABASE_URL = "sqlite:///./predictions.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    message = Column(String, index=True)
    spam = Column(Integer)
    probability = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class UserQuota(Base):
    __tablename__ = "user_quotas"
    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, unique=True, index=True)
    requests_count = Column(Integer, default=0)
    last_reset = Column(DateTime, default=datetime.utcnow)

class Stats(Base):
    __tablename__ = "stats"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow)
    total_predictions = Column(Integer, default=0)
    spam_count = Column(Integer, default=0)
    ham_count = Column(Integer, default=0)

Base.metadata.create_all(bind=engine)

# Modèles Pydantic
class Message(BaseModel):
    message: str

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=4)

class UserLogin(BaseModel):
    username: str
    password: str

# Configuration JWT
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

DAILY_QUOTA = 100  # Nombre maximum de requêtes par jour
QUOTA_RESET_HOURS = 24  # Réinitialisation du quota toutes les 24 heures

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_default_user():
    db = SessionLocal()
    try:
        existing_user = db.query(User).filter(User.username == "dassise").first()
        if existing_user:
            db.delete(existing_user)
            db.commit()
        
        hashed_password = hash_password("1234")
        default_user = User(
            username="dassise",
            email="dassise@example.com",
            password=hashed_password
        )
        db.add(default_user)
        db.commit()
        logging.info("Default user 'dassise' created/updated successfully.")
    except Exception as e:
        logging.error(f"Error creating default user: {e}")
        db.rollback()
    finally:
        db.close()

create_default_user()

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/register")
def register(user: UserRegister):
    db = SessionLocal()
    try:
        existing_username = db.query(User).filter(User.username == user.username).first()
        if existing_username:
            raise HTTPException(
                status_code=400, 
                detail="Ce nom d'utilisateur est déjà pris"
            )

        existing_email = db.query(User).filter(User.email == user.email).first()
        if existing_email:
            raise HTTPException(
                status_code=400, 
                detail="Cet email est déjà utilisé"
            )

        hashed_password = hash_password(user.password)
        new_user = User(
            username=user.username, 
            email=user.email, 
            password=hashed_password
        )
        db.add(new_user)
        db.commit()
        logging.info(f"User {user.username} registered successfully.")
        return {"msg": "Inscription réussie"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Erreur lors de l'inscription"
        )
    finally:
        db.close()

@app.post("/login")
def login(user: UserLogin):
    db = SessionLocal()
    try:
        print(f"Tentative de connexion pour l'utilisateur: {user.username}")
        
        db_user = db.query(User).filter(
            (User.username == user.username) | (User.email == user.username)
        ).first()
        
        if not db_user:
            print(f"Utilisateur non trouvé: {user.username}")
            raise HTTPException(status_code=401, detail="Utilisateur non trouvé")
            
        password_valid = verify_password(user.password, db_user.password)
        print(f"Vérification du mot de passe: {'succès' if password_valid else 'échec'}")
        print(f"Mot de passe fourni: {user.password}")
        print(f"Mot de passe hashé en DB: {db_user.password}")
        
        if not password_valid:
            raise HTTPException(status_code=401, detail="Mot de passe incorrect")

        access_token = create_access_token(data={"sub": db_user.email})
        print(f"Connexion réussie pour {user.username}")
        return {"access_token": access_token, "token_type": "bearer"}
    
    except Exception as e:
        print(f"Erreur de connexion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

def check_and_update_quota(db: Session, user_email: str) -> bool:
    quota = db.query(UserQuota).filter(UserQuota.user_email == user_email).first()
    
    if not quota:
        quota = UserQuota(user_email=user_email)
        db.add(quota)
        db.commit()
        return True

    # Réinitialiser le quota si nécessaire
    if datetime.utcnow() - quota.last_reset > timedelta(hours=QUOTA_RESET_HOURS):
        quota.requests_count = 0
        quota.last_reset = datetime.utcnow()
        db.commit()
        return True

    # Vérifier le quota
    if quota.requests_count >= DAILY_QUOTA:
        return False

    # Incrémenter le compteur
    quota.requests_count += 1
    db.commit()
    return True

def get_influential_words(message: str, prediction: float) -> dict:
    """
    Analyse les mots qui influencent la prédiction
    """
    try:
        # Vectoriser le message
        message_vector = vectorizer.transform([message])
        feature_names = vectorizer.get_feature_names_out()
        
        # Obtenir les coefficients du modèle
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
        else:
            coefficients = model.feature_importances_

        # Calculer l'importance de chaque mot
        word_importance = {}
        non_zero_indices = message_vector.nonzero()[1]
        
        for idx in non_zero_indices:
            word = feature_names[idx]
            importance = coefficients[idx] * message_vector[0, idx]
            word_importance[word] = float(importance)

        # Trier les mots par importance
        sorted_words = sorted(word_importance.items(), 
                            key=lambda x: abs(x[1]), 
                            reverse=True)

        # Séparer les indicateurs positifs et négatifs
        spam_indicators = []
        ham_indicators = []

        for word, importance in sorted_words[:5]:  # Prendre les 5 mots les plus influents
            if importance > 0:
                spam_indicators.append({
                    "word": word,
                    "impact": abs(importance)
                })
            else:
                ham_indicators.append({
                    "word": word,
                    "impact": abs(importance)
                })

        return {
            "spam_indicators": spam_indicators,
            "ham_indicators": ham_indicators
        }
    except Exception as e:
        print(f"Erreur dans get_influential_words: {str(e)}")
        return {
            "spam_indicators": [],
            "ham_indicators": [],
            "error": str(e)
        }

@app.post("/predict", 
    response_model=dict,
    summary="Prédire si un message est un spam",
    description="Analyse un message et retourne la probabilité qu'il soit un spam")
async def predict(
    message: Message = Body(..., example={"message": "Win a free iPhone now!"}),
    user: str = Depends(get_current_user)
):
    """
    Analyse un message pour déterminer s'il s'agit d'un spam.
    
    Args:
        message: Le message à analyser
        user: L'utilisateur authentifié (automatique via le token)
    
    Returns:
        dict: Résultat de l'analyse contenant:
            - spam (bool): True si spam, False sinon
            - probability (float): Probabilité que ce soit un spam
            - analysis (dict): Mots-clés influençant la décision
    """
    db = SessionLocal()
    try:
        if not message.message:
            raise HTTPException(status_code=400, detail="No message provided")

        # Vérifier le quota
        try:
            if not check_and_update_quota(db, user):
                raise HTTPException(
                    status_code=429, 
                    detail="Quota journalier dépassé. Réessayez demain."
                )
        except Exception as e:
            print(f"Erreur de quota: {str(e)}")
            pass

        # Faire la prédiction
        message_vector = vectorizer.transform([message.message])
        prediction_proba = model.predict_proba(message_vector)
        spam_probability = float(prediction_proba[0][1])

        # Obtenir les mots influents
        try:
            influential_words = get_influential_words(message.message, spam_probability)
        except Exception as e:
            print(f"Erreur d'analyse des mots: {str(e)}")
            influential_words = {"spam_indicators": [], "ham_indicators": []}

        # Sauvegarder la prédiction
        prediction = Prediction(
            message=message.message,
            spam=int(spam_probability >= 0.5),
            probability=spam_probability
        )
        db.add(prediction)
        db.commit()

        return {
            "spam": bool(spam_probability >= 0.5),
            "probability": round(spam_probability, 4),
            "analysis": influential_words,
            "quota": {
                "remaining": DAILY_QUOTA - (db.query(UserQuota)
                    .filter(UserQuota.user_email == user)
                    .first().requests_count if db.query(UserQuota)
                    .filter(UserQuota.user_email == user).first() else DAILY_QUOTA)
            }
        }
    except Exception as e:
        print(f"Erreur générale dans predict: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/history")
def get_history(user: str = Depends(get_current_user)):
    db = SessionLocal()
    try:
        predictions = db.query(Prediction).all()
        return predictions
    finally:
        db.close()

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    db = SessionLocal()
    try:
        contents = await file.read()
        df = pd.read_csv(contents.decode('utf-8'))
        results = []

        for message in df['message']:
            message_vector = vectorizer.transform([message])
            prediction_proba = model.predict_proba(message_vector)
            spam_probability = float(prediction_proba[0][1])
            
            prediction = Prediction(
                message=message,
                spam=int(spam_probability >= 0.5),
                probability=spam_probability
            )
            db.add(prediction)
            results.append({
                "message": message,
                "spam": bool(spam_probability >= 0.5),
                "probability": round(spam_probability, 4)
            })

        db.commit()
        return results
    finally:
        db.close()

@app.get("/")
async def root():
    return RedirectResponse(url="/login.html")

# Définir les routes API AVANT de monter les fichiers statiques
@app.get("/api/stats")
async def get_stats(current_user: str = Depends(get_current_user)):
    print(f"Accès aux statistiques pour l'utilisateur: {current_user}")
    db = SessionLocal()
    try:
        print("Début de la récupération des statistiques")
        
        total_predictions = db.query(Prediction).count()
        print(f"Total des prédictions: {total_predictions}")
        
        spam_count = db.query(Prediction).filter(Prediction.spam == True).count()
        print(f"Nombre de spams: {spam_count}")
        
        today = datetime.utcnow().date()
        predictions_today = db.query(Prediction).filter(
            func.date(Prediction.timestamp) == today
        ).count()
        print(f"Prédictions aujourd'hui: {predictions_today}")
        
        response_data = {
            "total_predictions": total_predictions,
            "spam_count": spam_count,
            "ham_count": total_predictions - spam_count,
            "predictions_today": predictions_today,
            "dates": [],
            "predictions_count": []
        }
        
        print(f"Données de réponse: {json.dumps(response_data)}")
        return response_data

    except Exception as e:
        print(f"Erreur dans get_stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des statistiques: {str(e)}"
        )
    finally:
        db.close()

@app.get("/api/quota")
async def get_quota(current_user: str = Depends(get_current_user)):
    print(f"Accès au quota pour l'utilisateur: {current_user}")
    db = SessionLocal()
    try:
        quota = db.query(UserQuota).filter(UserQuota.user_email == current_user).first()
        
        if not quota:
            return {"remaining": DAILY_QUOTA}
        
        if datetime.utcnow() - quota.last_reset > timedelta(hours=QUOTA_RESET_HOURS):
            quota.requests_count = 0
            quota.last_reset = datetime.utcnow()
            db.commit()
            return {"remaining": DAILY_QUOTA}
        
        remaining = max(0, DAILY_QUOTA - quota.requests_count)
        print(f"Quota restant: {remaining}")
        return {"remaining": remaining}

    except Exception as e:
        print(f"Erreur dans get_quota: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération du quota: {str(e)}"
        )
    finally:
        db.close()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5600)