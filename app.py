from flask import Flask, render_template, request
import utils
import pickle
import numpy as np
app = Flask(__name__)
model=pickle.load(open("iris_model.pkl","rb"))
scaler = pickle.load(open("scale.pkl","rb")) 

@app.route('/') 
def home(): 
    return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST']) 
def predict():
    if request.method == 'POST': 
        sepal_length = request.form.get('sepal_length')
        sepal_width = request.form.get('sepal_width')
        petal_length = request.form.get('petal_length')
        petal_width = request.form.get('petal_width') 
        X = [sepal_length,sepal_width,petal_length,petal_width]
        X = np.array([X])
        X=scaler.transform(X)
        Y_pred1 = model.predict(X)[0]
    # prediction = utils.preprocessdata(sepal_length,sepal_width, petal_length, petal_width) 

    return render_template('predict.html', prediction=Y_pred1) 
if __name__ == '__main__': 
    app.run(debug=True) 