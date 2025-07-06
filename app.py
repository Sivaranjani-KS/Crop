from app import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
app = Flask(__name__, template_folder='../templates')
import os
stacking_model = joblib.load(r'C:\\Project\\coding\\stacking_model .pkl')
preprocessor = joblib.load(r'C:\\Project\\coding\\preprocessor .pkl')
df = pd.read_csv("C:\\Project\\coding\\Tamilnadu agriculture yield data.csv")

def predict_production(crop_year, crop, season, district_name, area):
    input_df = pd.DataFrame({'District_Name': [district_name], 'Crop_Year': [crop_year], 'Season': [season], 'Crop': [crop], 'Area': [area]})
    input_transformed = preprocessor.transform(input_df)
    log_prediction = stacking_model.predict(input_transformed)
    final_prediction = np.expm1(log_prediction)
    return f"Predicted Production: {final_prediction[0]:.2f} tons"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        crop_year = int(request.form["crop_year"])
        crop = request.form["crop"]
        season = request.form["season"]
        district_name = request.form["district_name"]
        area = float(request.form["area"])

        prediction = predict_production(crop_year, crop, season, district_name, area)
        return render_template("crop.html", prediction=prediction)

    return render_template("crop.html", prediction="")

if __name__ == "_main_":
    app.run(debug=True)
