from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the Random Forest Classifier model
model = pickle.load(open("mysaved_mdrecommen_pickle.pkl", "rb"))

# Define the route for the home page
@app.route("/", methods = ["GET","POST"])
def home():
    return render_template("Input_2.html")

# Define the route for the prediction page
@app.route("/cropRecommender", methods=["GET","POST"])
def predict():
    #render_template('Input.html')
    # Get the symptoms from the request form
    # symptoms = request.form.to_dict()
    # Handle the form submission
    if request.method == "POST":
        # Get the list of selected features
        # selected_features = request.form.getlist('weight_loss')
        features = []
        feature = request.form.get('N')
        features.append(feature)

        feature = request.form.get('P')
        features.append(feature)

        feature = request.form.get('K')
        features.append(feature)

        feature = request.form.get('temperature')
        features.append(feature)

        feature = request.form.get('humidity')
        features.append(feature)

        feature = request.form.get('ph')
        features.append(feature)

        feature = request.form.get('rainfall')
        features.append(feature)

        # for i in range(1,133):
        #     feature = request.form.get('weight_loss')
        #     features.append(feature)
        # feature1 = request.form.get('weight_loss')
        # feature2 = request.form.get('weight_loss')
        # feature3 = request.form.get('weight_loss')
        # feature4 = request.form.get('weight_loss')

        # Make predictions based on the selected features
        # features = [feature1, feature2, feature3, feature4]

        # Make predictions based on the selected features
        prediction = model.predict(pd.DataFrame(features).T)
        print(features)
        return prediction[0]
    # Create a Pandas DataFrame from the symptoms
    # input_data = pd.DataFrame(symptoms, index=[0])
    # Convert the symptoms to numeric values (0 or 1)
    # input_data = input_data.apply(pd.to_numeric)
    # Make the prediction using the loaded model
    # prediction = model.predict(input_data)
    # Return the predicted disease prognosis
    #return jsonify({"prognosis": prediction[0]})
    # return prediction[0]

if __name__ == "__main__":
    app.run(debug=True, port=8080)
