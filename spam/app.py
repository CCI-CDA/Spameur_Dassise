from flask import Flask, render_template, request
from model import predict

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Récupérer le texte du formulaire
        message = request.form["message"]
        
        # Prédire si c'est un spam ou non
        result = predict(message)
        
        # Passer à la page HTML avec le résultat et le message
        return render_template("index.html", result=result, message=message)

    # Si la méthode est GET, on rend la page sans résultat
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
