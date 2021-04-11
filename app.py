import numpy as np
import os
import pandas
import sys
from flask import Flask, flash, request, redirect, url_for, session
from flask import send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
from functions.datscan_predict import datscan_predict
from functions.datscan_explain import datscan_explain

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './files/'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploade_speech', methods = ['GET', 'POST'])
def upload_speech():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
      n1 = request.form['num1']
      n2 = request.form['num2']
      n3 = request.form['num3']
      n4 = request.form['num4']
      n5 = request.form['num5']
      l = [n1,n2,n3,n4,n5]
      n = np.array(l)
      print(n)
      str1 = ''.join(l)
      return 'file uploaded successfully' + str1
   elif request.method == 'GET':
      return render_template('upload_speech.html')
      
		

@app.route('/upload_datscan', methods=['GET', 'POST'])
def upload_datscan():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            global file_path
            file_path = file.save(os.path.join("./files/datscans", filename))
            return redirect(url_for('upload_datscan', filename=filename))
    return render_template('upload_datscan.html')


@app.route('/datscan_predict', methods=['GET', 'POST'])
def predict_datscan():
    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            global file_path
            file_path = file.save(os.path.join("./files/datscans", filename))

        file_path = os.path.join("./files/datscans/", filename)
        hasPD = datscan_predict(file_path)
        datscan_explain(file_path)
        print(hasPD)
        return render_template('datscan_output.html', hasPD = hasPD)
    elif request.method == 'GET':
        scans = os.listdir('./files/datscans')
        return render_template('datscan_form.html',scans=scans)



if __name__ == '__main__':
    app.run(debug = True)
