from functools import reduce
import json
import os
from random import randint,choice
import string
import sys 
from flask import Flask,render_template,request,redirect,url_for,make_response,send_file
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
import numpy as np
import requests
from crawler.JobSpider import JobSpider
tf.enable_eager_execution()
app = Flask(__name__)
def loadNN(savedModelPath):
    model = load_model(savedModelPath)
    model.build(tf.TensorShape([1,None]))
    return model 
def generate_text(model,start_str,charToIdx,idxToChar):
    charCount = 300
    input_eval = [charToIdx[c] for c in start_str]
    input_eval = tf.expand_dims(input_eval,0)
    text_generated = []
    temperature = 0.7 
    model.reset_states()
    for i in range(charCount):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)
        predictions = predictions / temperature 
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id],0)
        text_generated.append(idxToChar[predicted_id])
    return ''.join(text_generated)
def loadModelParams(paramFilePath):
    modelParams = None
    with open(paramFilePath,"r") as f:
        modelParams = json.load(f)
    modelParams["idxToChar"] = {int(idx):c for idx,c in modelParams["idxToChar"].items()}
    return modelParams
@app.route("/")
def index():
    script = url_for(".static",filename = "js/script.js")
    return render_template("index.html",script = script)
@app.route("/test",methods=["POST"])
def test(): 
    if(request.form["seed"] == ""):
        seed = choice(string.ascii_letters) 
    else:
        jobPostUrl = request.form["seed"]
        spider = JobSpider(1,"data/",jobPostUrl)
        seed = spider.getJobAsDict()["body"]
    model = loadNN("./models/RNNModel.h5")
    modelParams = loadModelParams("./models/modelparams.json")
    haha = generate_text(model,seed,modelParams["charToIdx"],modelParams["idxToChar"])
    print(haha,file=sys.stderr)
    return redirect(url_for("confirm",body=haha))
@app.route("/confirm")
def confirm():
    return render_template("confirm.html",body=request.args.get("body"))
@app.route("/send",methods=["POST"])
def send():
    jobTitle = request.form["body"]
    print(request.form,file=sys.stderr)
    headers = {"Authorization": "Bearer 128475ccf2cfadbe56a78fac4ea1a9291d0a9f617edd17d3dddd7395779c03f62391bf8e9057af957ff412fb77f871246911612131e8dfaa55201c244b236eda","Content-Type":"application/json"}

    payload = json.JSONEncoder().encode({"title":jobTitle,"body":jobTitle})
    r = requests.post("https://hackicims.com/api/v1/companies/103/jobs",data=payload,headers=headers)
    print(r.status_code,file=sys.stderr)
    return redirect(url_for("index"))
@app.route("/haha")
def testing():
    return url_for(".static",filename="chaebae.jpg")
if __name__ == "__main__":
    app.run(host="127.0.0.1",debug=True,port="8000")
