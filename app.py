from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/scans'
@app.route('/upload')
def upload_file():
   return render_template('form.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
    if request.method == 'POST':
       f = request.files['file']
       f.save(secure_filename(f.filename))
       n1 = request.form['num1']
       n2 = request.form['num2']
       n3 = request.form['num3']
       n4 = request.form['num4']
       n5 = request.form['num5']
       l = [n1,n2,n3,n4,n5]
       n = np.array(l)
       
       print (n)
       str1 = ''.join(l)
    return 'file uploaded successfully' + str1
		
if __name__ == '__main__':
   app.run(debug = True)