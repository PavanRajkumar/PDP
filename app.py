import numpy as np
import os, pandas, sys
from flask import Flask, flash, request, redirect, url_for, session
from flask import send_from_directory, render_template, jsonify
from werkzeug.utils import secure_filename
from functions.datscan_predict import datscan_predict
from functions.datscan_explain import datscan_explain
from functions.db_functions import writeToDB
from speech_diagnosis.src.Audio_Controller import Audio_Controller
import datetime, pytz


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './files/'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/history')
def history():
    return render_template('history.html')


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
      print(f)
      str1 = ''.join(l)
      return 'file uploaded successfully' + str1
   elif request.method == 'GET':
      return render_template('upload_speech.html')


@app.route('/form_upload', methods=['GET', 'POST'])
def form_upload():

    if request.method == "POST":
        first = request.form['FirstName']
        last = request.form['LastName']
        age = request.form['age']
        gender = request.form['gender']

        file_paths = {'datscan': None, 'speech': None}
        file_paths = get_file_path(request=request)
        print("FILE_PATHS",file_paths)


        if file_paths['datscan'] != None:
            hasPDdatscan = datscan_predict(file_paths['datscan'])
        else:
            hasPDdatscan = None
        # datscan_explain(file_path)

        if file_paths.get('speech',None) != None:
            audio = Audio_Controller(file_paths['speech'])
            audio.process_audio()
            hasPDspeech = audio.predict_PD_diagnosis(model_name="RF")
        else:
            hasPDspeech = None

        current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))

        data = {
            'first': first,
            'last': last,
            'age': age,
            'gender': gender,
            'hasPDdatscan': hasPDdatscan,
            'datscanPath': file_paths['datscan'],
            'hasPDspeech': hasPDspeech,
            'speechPath': file_paths['speech'],
            'predictTime': current_time
        }

        # writeToDB(data)

        return render_template('datscan_output.html', data=data)

    elif request.method == 'GET':
        scans = os.listdir('files/datscan')
        return render_template('datscan_form.html', scans=scans)


def get_file_path(request):

    file_paths = {"datscan":None, "speech":None}

    for file_type in file_paths.keys():

        if file_type not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files[file_type]

        if file.filename == '':
            flash('No selected file')
            # return redirect(request.url)

        filename = None
        file_path = None

        if file:
            filename = secure_filename(file.filename)
            print("FILENAME", filename)
            file.save(os.path.join("../PDP/files/{0}/".format(file_type), filename))
            file_path = os.path.join("../PDP/files/{}".format(file_type), filename)

        file_paths[file_type] = file_path

    return file_paths


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug = True)
