

from flask import Flask,request,render_template
import numpy as np

import joblib



model=joblib.load('rfccJoblib.pkl')
ms=joblib.load('ssJoblib.pkl')

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list =np.array([[N, P, K, temp, humidity, ph, rainfall]]) 
    proba=model.predict_proba(feature_list)[0]
    top3_indices=np.argsort(proba)[-3:][::-1]

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}
    top3_crops=[crop_dict[i] for i in top3_indices]
    
    return render_template('index.html',prediction = top3_crops)


# python main
if __name__ == "__main__":
    app.run(debug=True)