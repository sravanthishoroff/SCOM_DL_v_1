import os, re, json, shutil
from flask import Flask, redirect , url_for, render_template , request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from alarms import scom_alarm
import zipfile


app = Flask(__name__)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'

@app.route('/')
def index():
    return render_template('flask.html')

@app.route('/uploads', methods=['POST'])
def upload_file():
    UPLOAD_FOLDER  = os.getcwd()+'\output'

    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    if os.path.exists(UPLOAD_FOLDER) and os.path.isdir(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.mkdir(UPLOAD_FOLDER)

    upload_data = request.files['myFile']
        
    # calling function for predicting image
    result = scom_alarm(upload_data)

    shutil.move('false_alarm_report.xls', UPLOAD_FOLDER)

    shutil.make_archive('result', 'zip', 'output')

    return result

@app.route('/download', methods =['GET'])
def download_all():
#     zipf = zipfile.ZipFile('result_file.zip','w', zipfile.ZIP_DEFLATED)
#     for root,dirs, files in os.walk(UPLOAD_FOLDER):
#         for file in files:
#             zipf.write('result_file/'+file)
# #     zipf.close()
    return send_file('result.zip', as_attachment=True, cache_timeout=-1)


if __name__ == '__main__':
    app.run()