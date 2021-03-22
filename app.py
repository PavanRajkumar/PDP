import numpy as np
import os
import pandas
import sys
from flask import Flask, flash, request, redirect, url_for
from flask import send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename


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
            file_path = file.save(os.path.join("./files", filename))
            return redirect(url_for('upload_datscan', filename=filename))
    return render_template('upload_datscan.html')


if __name__ == '__main__':
   app.run(debug = True)