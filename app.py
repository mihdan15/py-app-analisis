from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # This will allow cross-origin requests

# Load the models
model1 = joblib.load('model1.sav')
model2 = joblib.load('model2.sav')
model3 = joblib.load('model3.sav')

@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No input text provided'}), 400

    try:
        pred1 = model1.predict([text])[0]
        pred2 = model2.predict([text])[0]
        pred3 = model3.predict([text])[0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    result = {
        'DecisionTreeClassifier': pred1,
        'MultinomialNB': pred2,
        'RandomForestClassifier': pred3
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
