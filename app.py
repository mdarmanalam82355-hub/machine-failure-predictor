from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

data = {
    "temperature": [30,40,50,60,70,80,90],
    "pressure": [100,110,120,130,140,150,160],
    "vibration": [10,15,20,25,30,35,40],
    "failure": [0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)
X = df[["temperature","pressure","vibration"]]
y = df["failure"]

model = LogisticRegression()
model.fit(X,y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = model.predict([[data['temperature'], data['pressure'], data['vibration']]])
    return jsonify({"prediction": int(result[0])})

app.run(host='0.0.0.0', port=5000)
