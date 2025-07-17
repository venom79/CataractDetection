from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("./model/cataract_model.keras")

# Home Page
@app.route("/")
def home():
    return render_template("home.html")

# Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    image_file = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join("static", "temp.jpg")
            file.save(file_path)
            image_file = "temp.jpg"

            # Preprocess image
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            result = model.predict(img_array)[0][0]
            prediction = "Cataract" if result < 0.5 else "Normal"

    return render_template("predict.html", prediction=prediction, image_file=image_file)

# Doctors Page
@app.route("/doctors")
def doctors():
    doctors = [
        {
            "name": "Dr. Ramesh Kulkarni",
            "address": "Vision Plus Eye Clinic, MG Road, Panaji",
            "phone": "+91 9876543210",
            "timings": "Mon-Sat: 9 AM – 6 PM"
        },
        {
            "name": "Dr. Neha Mehta",
            "address": "Shree Netralaya, Vasco da Gama",
            "phone": "+91 8765432109",
            "timings": "Mon-Fri: 10 AM – 5 PM"
        },
        {
            "name": "Dr. Anil Patil",
            "address": "Goa Eye Hospital, Mapusa",
            "phone": "+91 9123456780",
            "timings": "Mon-Sat: 8 AM – 2 PM"
        },
        {
            "name": "Dr. Sneha Desai",
            "address": "Desai Eye Care, Margao",
            "phone": "+91 9988776655",
            "timings": "Tue-Sun: 11 AM – 7 PM"
        },
        {
            "name": "Dr. Vivek Shenoy",
            "address": "Shenoy Eye Institute, Porvorim",
            "phone": "+91 9001122334",
            "timings": "Mon-Sat: 10 AM – 6 PM"
        }
    ]

    return render_template("doctors.html", doctors=doctors)


if __name__ == "__main__":
    app.run(debug=True)
