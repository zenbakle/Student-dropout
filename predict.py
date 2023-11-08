import pickle
from flask import Flask ,request, jsonify

app = Flask(__name__)
#model name

with open("dropout1.bin" , "rb") as f_in:
   dv,model = pickle.load(f_in)


@app.route("/predict",methods=["POST"])
def predict():
    student = request.get_json()
    X = dv.transform([student])
    pred = model.predict_proba(X)[0,1]
    churn = pred >= 0.5
    result = {
        "Dropout probability": float(pred),
        "Student will Dropout?": bool(churn)
    }
    return jsonify(result)


if __name__=="__main__":
    app.run(debug=True)