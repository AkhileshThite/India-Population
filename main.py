# Importing necessary libraries.
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

# Making an instance of Flask
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# This will head us to the main index.html file.
@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    y_pred = model.predict(final_features)
    output = '%.2f' % y_pred
    msg = request.form['year']

    return render_template('index.html', final_prediction=f'Population for the year {msg}: {output}' + ' million')


# To run the Flask app on server.
if __name__ == "__main__":
    app.run(debug=True)
