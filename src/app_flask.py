from flask import Flask, request, render_template
import json
from xgboost import XGBRegressor


app = Flask(__name__)
model = XGBRegressor()
model.load_model('../models/xgboost_optimizado.json')

with open('../data/processed/dic_cl.json','r', encoding='utf-8') as archivo:
    dic_cl = json.load(archivo)
with open('../data/processed/dic_el.json','r', encoding='utf-8') as archivo:
    dic_el = json.load(archivo)
with open('../data/processed/dic_et.json','r', encoding='utf-8') as archivo:
    dic_et = json.load(archivo)
with open('../data/processed/dic_er.json','r', encoding='utf-8') as archivo:
    dic_er = json.load(archivo)
with open('../data/processed/dic_cs.json','r', encoding='utf-8') as archivo:
    dic_cs = json.load(archivo)

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        
        # Obtain values from form
        val1 = float(request.form["val1"])#work_year
        val2 = request.form["val2"]#experience_level
        val3 = request.form["val3"]#employment_type
        val4 = float(request.form["val4"])#remote_ratio
        val5 = request.form["val5"]#company_size
        val6 = request.form["val6"]#employee_residence_num
        val7 = request.form["val7"]#company_location
        
        data = [[val1, dic_el[val2], dic_et[val3], val4,dic_cs[val5],dic_er[val6],dic_cl[val7]]]
        prediction = str(model.predict(data)[0])
        pred_class = prediction
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)