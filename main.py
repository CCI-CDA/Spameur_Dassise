from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import jwt
import datetime
from fastapi.responses import JSONResponse

app = FastAPI()

# Clé secrète pour générer les jetons JWT
SECRET_KEY = "votre_clé_secrète"
ALGORITHM = "HS256"

# Base de données simulée des utilisateurs
fake_users_db = {
    "user1": {"username": "user1", "password": "password123"},
}

# Modèle de données pour l'authentification
class UserLogin(BaseModel):
    username: str
    password: str

# Fonction pour créer un jeton JWT
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Route pour se connecter
@app.post("/login")
def login(user: UserLogin):
    user_data = fake_users_db.get(user.username)
    if user_data and user_data["password"] == user.password:
        token = create_access_token(data={"sub": user.username})
        return JSONResponse(content={"token": token})
    raise HTTPException(status_code=401, detail="Identifiants incorrects")

# Page de connexion (redirige vers la page HTML)
@app.get("/login.html")
def login_page():
    return RedirectResponse(url='/login.html')

# Page d'index (accès protégé avec le jeton JWT)
@app.get("/index.html")
def index_page():
    return RedirectResponse(url='/index.html')
