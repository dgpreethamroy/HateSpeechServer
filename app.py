from flask import Flask,request,render_template
app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def homme():
    if request.method == 'GET':
        return 'Hello, World! I think I am ready to be a full stack developer!'
    else:
        return 'You should not be here. Go back to the home page'

@app.route('/hello/<name>',methods=['GET','POST'])
def hello(name):
    return "Hello, {}!".format(request.json["Query"])
if __name__ == '__main__':
    app.run()

import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


@app.route('/predict',methods=['POST'])
def HSD():
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("static/vectorizer_HSD.pkl", "rb")))
    loaded_model = pickle.load(open('static/trained_HSD.sav', 'rb'))
    user = request.json["Query"]
    data = loaded_vec.transform(np.array([user]))
    result = loaded_model.predict(data)
    return result[0]
