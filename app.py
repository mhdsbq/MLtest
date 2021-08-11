# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from numpy.lib.type_check import imag
from werkzeug.utils import secure_filename
import os

from predictor import predictor

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = predictor(file_path)
    return result


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

