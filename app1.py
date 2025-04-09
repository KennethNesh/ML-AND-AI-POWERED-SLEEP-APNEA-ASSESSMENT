from flask import Flask, request, render_template_string
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your pre-trained neural network model
model = load_model('sleep_apnea_model1.h5')

# HTML template
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sleep Apnea Risk Assessment</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #3B82F6;
            --secondary-color: #10B981;
            --background-color: #F3F4F6;
            --text-color: #1F2937;
            --error-color: #EF4444;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
            line-height: 1.6;
        }
        .container {
            width: 100%;
            max-width: 700px;
            margin: 2rem auto;
            padding: 2rem;
            background-color: white;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            border-radius: 12px;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
            color: var(--primary-color);
        }
        .form-group {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        input, select {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid #D1D5DB;
            border-radius: 8px;
            font-size: 1rem;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary-color);
        }
        .submit-btn {
            width: 100%;
            padding: 1rem;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
        }
        .submit-btn:hover {
            background-color: #2563EB;
        }
        .error-message {
            color: var(--error-color);
            font-size: 0.875rem;
            margin-top: 0.5rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sleep Apnea Risk Assessment</h1>
            <p>Fill out the form to assess your potential risk</p>
        </div>
        {% if error %}
            <div class="error-message">{{ error }}</div>
        {% endif %}
        <form method="post" action="/">
            <div class="form-group">
                <label for="age">Age (years):</label>
                <input type="number" id="age" name="age" min="30" max="70" step="1" value="{{ form_data.age|default('') }}" required>
            </div>

            <div class="form-group">
                <label for="sex">Sex:</label>
                <select id="sex" name="sex" required>
                    <option value="">Select Sex</option>
                    <option value="male" {% if form_data.sex == 'male' %}selected{% endif %}>Male</option>
                    <option value="female" {% if form_data.sex == 'female' %}selected{% endif %}>Female</option>
                </select>
            </div>

            <div class="form-group">
                <label for="waist_hip_ratio">Waist-Hip Ratio:</label>
                <input type="number" id="waist_hip_ratio" name="waist_hip_ratio" step="0.01" min="0.5" max="1.5" value="{{ form_data.waist_hip_ratio|default('') }}" required placeholder="e.g., 0.85">
            </div>

            <div class="form-group">
                <label for="active_smoking">Active Smoking:</label>
                <select id="active_smoking" name="active_smoking" required>
                    <option value="">Select</option>
                    <option value="yes" {% if form_data.active_smoking == 'yes' %}selected{% endif %}>Yes</option>
                    <option value="no" {% if form_data.active_smoking == 'no' %}selected{% endif %}>No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="passive_smoking">Exposure to Passive Smoke:</label>
                <select id="passive_smoking" name="passive_smoking" required>
                    <option value="">Select</option>
                    <option value="yes" {% if form_data.passive_smoking == 'yes' %}selected{% endif %}>Yes</option>
                    <option value="no" {% if form_data.passive_smoking == 'no' %}selected{% endif %}>No</option>
                </select>
            </div>

            <div class="form-group">
                <label for="alcohol">Alcohol Consumption (drinks per week):</label>
                <input type="number" id="alcohol" name="alcohol" min="0" max="14" step="1" value="{{ form_data.alcohol|default('') }}" required>
            </div>

            <div class="form-group">
                <label for="physical_activity">Physical Activity (0-10):</label>
                <input type="number" id="physical_activity" name="physical_activity" min="0" max="10" step="1" value="{{ form_data.physical_activity|default('') }}" required placeholder="0 = sedentary, 10 = very active">
            </div>

            <div class="form-group">
                <label for="diet_quality">Diet Quality (0-10):</label>
                <input type="number" id="diet_quality" name="diet_quality" min="0" max="10" step="1" value="{{ form_data.diet_quality|default('') }}" required placeholder="0 = poor, 10 = excellent">
            </div>

            <div class="form-group">
                <label for="mental_health">Mental Health Stress Level (0-20):</label>
                <input type="number" id="mental_health" name="mental_health" min="0" max="20" step="0.1" value="{{ form_data.mental_health|default('') }}" required placeholder="0 = best, higher = more stress">
            </div>

            <button type="submit" class="submit-btn">Assess My Risk</button>
        </form>
    </div>
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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #3B82F6;
            --secondary-color: #10B981;
            --background-color: #F3F4F6;
            --text-color: #1F2937;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--background-color);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            color: var(--text-color);
        }
        .result-container {
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            text-align: center;
            max-width: 500px;
            width: 100%;
        }
        .result-title {
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        .risk-category {
            font-size: 2rem;
            font-weight: 600;
            margin: 1rem 0;
        }
        .back-link {
            display: inline-block;
            margin-top: 1rem;
            padding: 0.75rem 1.5rem;
            background-color: var(--secondary-color);
            color: white;
            text-decoration: none;
            border-radius: 8px;
        }
        .back-link:hover {
            background-color: #059669;
        }
    </style>
</head>
<body>
    <div class="result-container">
        <h2 class="result-title">Sleep Apnea Risk Assessment</h2>
        <div class="risk-category">{{ result }}</div>
        <p>Please consult with a healthcare professional for a comprehensive evaluation.</p>
        <a href="/" class="back-link">Perform Another Assessment</a>
    </div>
</body>
</html>
"""

def map_to_category(probability):
    """
    Map the probability (0-1) to a categorical risk level.
    """
    percent = probability * 100
    if percent < 30:
        return "Low Risk"
    elif percent < 70:
        return "Moderate Risk"
    else:
        return "High Risk"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            # Validate and convert inputs
            age = float(form_data.get("age", 0))
            if not (30 <= age <= 70):
                raise ValueError("Age must be between 30 and 70")
            sex = form_data.get("sex")
            if sex not in ["male", "female"]:
                raise ValueError("Please select a valid sex")
            waist_hip_ratio = float(form_data.get("waist_hip_ratio", 0))
            if not (0.5 <= waist_hip_ratio <= 1.5):
                raise ValueError("Waist-Hip Ratio must be between 0.5 and 1.5")
            active_smoking = form_data.get("active_smoking")
            if active_smoking not in ["yes", "no"]:
                raise ValueError("Please select active smoking status")
            passive_smoking = form_data.get("passive_smoking")
            if passive_smoking not in ["yes", "no"]:
                raise ValueError("Please select passive smoking status")
            alcohol = float(form_data.get("alcohol", 0))
            if not (0 <= alcohol <= 14):
                raise ValueError("Alcohol consumption must be between 0 and 14")
            physical_activity = float(form_data.get("physical_activity", 0))
            if not (0 <= physical_activity <= 10):
                raise ValueError("Physical activity must be between 0 and 10")
            diet_quality = float(form_data.get("diet_quality", 0))
            if not (0 <= diet_quality <= 10):
                raise ValueError("Diet quality must be between 0 and 10")
            mental_health = float(form_data.get("mental_health", 0))
            if not (0 <= mental_health <= 20):
                raise ValueError("Mental health stress level must be between 0 and 20")
            
            # Convert categorical values to numerical equivalents
            sex_num = 1 if sex == "male" else 0
            active_smoking_num = 1 if active_smoking == "yes" else 0
            passive_smoking_num = 1 if passive_smoking == "yes" else 0
            
            # Create feature vector
            features = np.array([[age, sex_num, waist_hip_ratio, active_smoking_num,
                                  passive_smoking_num, alcohol, physical_activity,
                                  diet_quality, mental_health]])
            
            # Model prediction
            prediction = model.predict(features)[0][0]
            category = map_to_category(prediction)
            
            return render_template_string(result_html, result=category)
        except ValueError as e:
            return render_template_string(index_html, error=str(e), form_data=form_data)
        except Exception as e:
            return render_template_string(index_html, error="An error occurred while processing your data", form_data=form_data)
    return render_template_string(index_html, form_data={})

if __name__ == "__main__":
    app.run(debug=True)
