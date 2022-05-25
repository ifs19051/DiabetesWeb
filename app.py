#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/back')
def back():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')
    
@app.route('/Pregnancies')
def Pregnancies():
    return render_template('Pregnancies.html')
    
@app.route('/Glucose')
def Glucose():
    return render_template('Glucose.html')
    
@app.route('/BloodPressure')
def BloodPressure():
    return render_template('BloodPressure.html')
    
@app.route('/SkinThickness')
def SkinThickness():
    return render_template('SkinThickness.html')
    
@app.route('/Insulin')
def Insulin():
    return render_template('Insulin.html')
    
@app.route('/BMI')
def BMI():
    return render_template('BMI.html')
    
@app.route('/DiabetesPedigreeFunction')
def DiabetesPedigreeFunction():
    return render_template('DiabetesPedigreeFunction.html')
    
@app.route('/Age')
def Age():
    return render_template('Age.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    #return render_template('index.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(prediction))
    if(prediction == 0):
        return render_template('berhasil.html')
    return render_template('gagal.html')    
    #return render_template('index.html', prediction_text='CO2 Emission of the vehicle is :{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)