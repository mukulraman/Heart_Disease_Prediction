import pandas as pd
from flask import Flask, request, jsonify
import joblib

from DataCleaning import data_cleaning

# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = joblib.load('models\Model_Classifier_Heart.pkl')
features = joblib.load('models\Features_Columns.pkl')


# Define a route for the prediction endpoint
@app.route('/predict', methods=["POST"])

def predict():
    data = request.json
    input_data = pd.DataFrame(data['data'])
    print("Input DataFrame Columns:", input_data.columns)  # Debugging line

    try:
        X_input = data_cleaning(input_data)
        X_input = X_input.reindex(columns=features, fill_value=0)

    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 400

    predictions = model.predict(X_input)
    return jsonify({'predictions': predictions.tolist()})

# Run the app if executed directly
if __name__ == "__main__":
    app.run(port=5001)
