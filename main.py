import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
model = pickle.load(open('xgboost_model.pkl', 'rb'))
print("hello ")
app1 = Flask(__name__)
@app1.route("/")#end point provide kar raha hu (url ke last me / hona cha)
def index():
    return render_template("/web.html")



@app1.route("/predict", methods =['GET' , 'POST'])
def sense():
   # fuel = request.form.get("fuel_type")

    data1 = {
        'Year': request.form.get("manufacture_year"),
        'Kilometers_Driven': request.form.get("kilometer_driven"),
        'Fuel_Type': request.form.get("fuel_type"),
        'Transmission': request.form.get("transmission"),
        'Owner_Type': request.form.get("ownertype"),
        'Mileage': request.form.get("mileage"),
        'Engine': request.form.get("engine"),
        'Power': request.form.get("power"),
        'Seats': request.form.get("seats"),
        'Brand_Category': request.form.get("car_brand")}
    print(data1)
    
# Wrap scalar values in lists
    data1 = {key: [value] for key, value in data1.items()}
    given = pd.DataFrame(data1)
    print(given.info())

    given['Year']= given['Year'].astype(int)
    given['Kilometers_Driven']= given['Kilometers_Driven'].astype(int)
    given['Fuel_Type']= given['Fuel_Type'].astype(int)
    given['Transmission']= given['Transmission'].astype(int)
    given['Owner_Type']= given['Owner_Type'].astype(int)
    given['Seats']= given['Seats'].astype(int)
    given['Brand_Category']= given['Brand_Category'].astype(int)
    given['Mileage']= given['Mileage'].astype(float)
    given['Power']= given['Power'].astype(float)
    given['Engine']= given['Engine'].astype(float)
    answer= np.expm1(model.predict(given))
    answer = answer.round(2)
    print(answer)
   
    return render_template("/predict.html" , prediction_text=f"Your car can sell around {answer}")
if __name__ == "__main__":
    app1.run(debug= True)