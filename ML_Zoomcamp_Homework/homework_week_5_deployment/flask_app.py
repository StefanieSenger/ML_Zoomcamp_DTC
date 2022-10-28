from flask import Flask
from flask import request
from flask import jsonify
import pickle

# Getting Model and DictVectorizer from pickle files
dv_file = 'dv.bin'
model_file = 'model1.bin'

with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)


# Defining Flask connection
app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    transformed_test_data = dv.transform([client])
    y_pred = model.predict_proba(transformed_test_data)[0,1]
    result = {'probability_of_getting_credit_card': float(y_pred), 'credit_card': bool(y_pred >= 0.5)}
    return jsonify(result)


#if __name__ == "__main__":
#    app.run(debug=True, host='0.0.0.0', port=9696)
