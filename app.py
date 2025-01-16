import logging
import pickle

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the spam detection model
try:
    with open('spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Load the vectorizer
try:
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info("Vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading vectorizer: {e}")
    raise

class Message(BaseModel):
    message: str

@app.post("/predict")
def predict(message: Message):
    if not message.message:
        raise HTTPException(status_code=400, detail="No message provided")

    try:
        # Transform the message into a vector
        message_vector = vectorizer.transform([message.message])
        logging.info("Message transformed into vector successfully.")
    except Exception as e:
        logging.error(f"Error transforming message: {e}")
        raise HTTPException(status_code=500, detail="Error processing message")

    try:
        # Predict if the message is spam
        prediction_proba = model.predict_proba(message_vector)
        spam_probability = float(prediction_proba[0][1])  # Convert to float
        logging.info(f"Prediction made successfully with probability: {spam_probability}")
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail="Error making prediction")

    return {
        "spam": bool(spam_probability >= 0.5),
        "probability": spam_probability.__round__(4)
    }

# Serve static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)