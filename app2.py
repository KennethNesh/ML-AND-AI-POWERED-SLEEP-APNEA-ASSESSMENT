from flask import Flask, request, render_template_string
import numpy as np
from tensorflow.keras.models import load_model
import google.generativeai as genai
import re

app = Flask(__name__)

# Load your pre-trained model
model = load_model('sleep_apnea_model1.h5')

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyAmx6bOG7QLzrS31x6HNn1g8QUtPHjGpH0"
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# HTML Templates
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Sleep Apnea Risk Assessment</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #f0f4ff; text-align: center; padding: 40px; }
        form { display: inline-block; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.1); text-align: left; width: 400px; }
        h1 { color: #4b0082; }
        input, select { margin-top: 10px; padding: 10px; width: 100%; border-radius: 6px; border: 1px solid #ccc; }
        button { background: #4b0082; color: white; padding: 12px; border: none; border-radius: 8px; width: 100%; font-weight: bold; margin-top: 15px; cursor: pointer; }
        button:hover { background: #3a006b; }
    </style>
</head>
<body>
    <h1>Sleep Apnea Risk Assessment</h1>
    <form method="post">
        Age: <input type="number" name="age" required>
        Sex:
        <select name="sex" required>
            <option value="male">Male</option>
            <option value="female">Female</option>
        </select>
        Waist-Hip Ratio: <input type="number" step="0.01" name="waist_hip_ratio" required>
        Active Smoking:
        <select name="active_smoking" required>
            <option value="yes">Yes</option>
            <option value="no">No</option>
        </select>
        Passive Smoking:
        <select name="passive_smoking" required>
            <option value="yes">Yes</option>
            <option value="no">No</option>
        </select>
        Alcohol (drinks/week): <input type="number" name="alcohol" required>
        Physical Activity (0-10): <input type="number" name="physical_activity" required>
        Diet Quality (0-10): <input type="number" name="diet_quality" required>
        Mental Health Stress (0-20): <input type="number" name="mental_health" required>
        <button type="submit">Assess My Risk</button>
    </form>
</body>
</html>
"""

result_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assessment Result</title>
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f5f5ff;
            padding: 30px;
        }
        .container {
            max-width: 800px;
            margin: auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .risk-box {
            background-color: #f0f0ff;
            padding: 40px;
            margin: 30px auto;
            font-size: 28px;
            font-weight: bold;
            border-radius: 12px;
            border: 2px solid #6a5acd;
            text-align: center;
            max-width: 400px;
        }
        .typing-section {
            white-space: pre-wrap;
            padding-left: 10px;
            margin-top: 30px;
            font-size: 16px;
            line-height: 1.8;
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        a {
            display: block;
            margin-top: 30px;
            text-align: center;
            color: #6a5acd;
            text-decoration: none;
            font-weight: bold;
        }
        a:hover {
            text-decoration: underline;
        }
        .powered {
            margin-top: 30px;
            text-align: center;
            color: #888;
            font-style: italic;
            font-size: 14px;
        }
    </style>
    <script>
        window.onload = function () {
            const text = document.getElementById('typing-text').textContent;
            document.getElementById('typing-text').textContent = '';
            let i = 0;
            const typingInterval = setInterval(() => {
                document.getElementById('typing-text').textContent += text[i];
                i++;
                if (i >= text.length) clearInterval(typingInterval);
            }, 10);
        };
    </script>
</head>
<body>
    <div class="container">
        <h1 style="text-align:center;">Sleep Apnea Risk Assessment Result</h1>
        <div class="risk-box">{{ risk_level }}</div>
        
        <div class="typing-section">
            <h2>Interpretation and Advice:</h2>
            <div id="typing-text">{{ gemini_advice }}</div>
        </div>
        
        <a href="/">Perform Another Assessment</a>
        <div class="powered">©KennethNesh</div>
    </div>
</body>
</html>
"""

def get_risk_level(score):
    if score < 0.33:
        return "Low Risk"
    elif score < 0.66:
        return "Mild Risk"
    else:
        return "High Risk"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form = request.form
        features = np.array([[ 
            float(form["age"]),
            1 if form["sex"] == "male" else 0,
            float(form["waist_hip_ratio"]),
            1 if form["active_smoking"] == "yes" else 0,
            1 if form["passive_smoking"] == "yes" else 0,
            float(form["alcohol"]),
            float(form["physical_activity"]),
            float(form["diet_quality"]),
            float(form["mental_health"])
        ]])
        prediction = float(model.predict(features)[0][0])
        risk_level = get_risk_level(prediction)

        prompt = f"""
        A user has been classified as having {risk_level.lower()} for sleep apnea based on several lifestyle factors.

        Please explain what this means in simple, encouraging terms and give personalized, practical tips they can follow. 
        Make sure the tone is warm and supportive.
        """

        gemini_response = model_gemini.generate_content(prompt)
        gemini_text = gemini_response.text.strip()

        return render_template_string(result_html, risk_level=risk_level, gemini_advice=gemini_text)

    return render_template_string(index_html)

if __name__ == "__main__":
    app.run(debug=True)
