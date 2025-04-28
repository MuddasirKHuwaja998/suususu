import os
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Path to the trained model
MODEL_PATH = "saved_model/model.h5"

# Load the trained model
model = load_model(MODEL_PATH)

# Class details in Italian with alternating details
class_details_alternate = {
    "Chronic Otitis Media": [
        {
            "description": "L'otite media cronica è un'infezione persistente dell'orecchio che può causare secrezione.",
            "causes": ["Infezioni ripetute dell'orecchio", "Funzione scarsa della tromba di Eustachio", "Allergie"]
        },
        {
            "description": "L'otite media cronica può causare sordità progressiva e dolore costante.",
            "causes": ["Uso di oggetti nell'orecchio", "Complicazioni di infezioni respiratorie", "Problemi immunologici"]
        },
    ],
    "Earwax Plug": [
        {
            "description": "Il tappo di cerume si verifica quando il cerume si accumula e blocca il canale uditivo.",
            "causes": ["Produzione eccessiva di cerume", "Pulizia impropria dell'orecchio", "Uso di auricolari o apparecchi acustici"]
        },
        {
            "description": "Un tappo di cerume può causare prurito, irritazione e perdita temporanea dell'udito.",
            "causes": ["Uso di cotton fioc", "Accumulo cronico di cera", "Anomalie del canale uditivo"]
        }
    ],
    "Myringosclerosis": [
        {
            "description": "La miringosclerosi comporta la formazione di cicatrici sul timpano, spesso a causa di infezioni ripetute.",
            "causes": ["Infezioni ripetute dell'orecchio", "Trauma all'orecchio", "Interventi chirurgici sull'orecchio"]
        },
        {
            "description": "La miringosclerosi può causare rigidità del timpano e perdita parziale dell'udito.",
            "causes": ["Esposizione a rumori forti", "Trattamenti medici invasivi", "Infiammazione cronica"]
        }
    ],
    "Normal": [
        {
            "description": "L'orecchio appare normale senza segni di infezione o anomalie.",
            "causes": ["Orecchio sano", "Nessun problema sottostante"]
        },
        {
            "description": "L'orecchio è in condizioni ottimali, senza segni di malattie.",
            "causes": ["Buona igiene auricolare", "Stile di vita sano"]
        }
    ]
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing function
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Prediction function
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = list(class_details_alternate.keys())[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

# Toggle flag for alternating details
toggle_flag = {key: 0 for key in class_details_alternate.keys()}

@app.route("/", methods=["GET", "POST"])
def index():
    global toggle_flag

    if request.method == "POST":
        # Check if a file is uploaded
        if "file" not in request.files:
            flash("Nessun file caricato!")
            return redirect(request.url)

        file = request.files["file"]

        # Handle single image upload
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_dir = os.path.join("static", "uploads")
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, filename)
            file.save(file_path)

            # Predict the uploaded image
            predicted_class, confidence = predict_image(file_path)

            # Toggle between the two sets of details
            current_flag = toggle_flag[predicted_class]
            details = class_details_alternate[predicted_class][current_flag]
            toggle_flag[predicted_class] = 1 - current_flag  # Flip the toggle

            return render_template(
                "index.html",
                uploaded_image=url_for("static", filename=f"uploads/{filename}"),
                prediction=predicted_class,
                confidence=f"{confidence:.2f}%",
                description=details["description"],
                causes=details["causes"]
            )

        else:
            flash("Tipo di file non valido! Carica un'immagine valida.")
            return redirect(request.url)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)