from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = './'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('Main.html')
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        ext = filename.rsplit('.', 1)[1]
        f.save(os.path.join(UPLOAD_FOLDER, filename))
        try:
            os.rename(UPLOAD_FOLDER + filename, UPLOAD_FOLDER +'1.' + ext)
        except:
            os.remove(UPLOAD_FOLDER + '1.' + ext)
            os.rename(UPLOAD_FOLDER + filename, UPLOAD_FOLDER +'1.' + ext)
        model = tf.keras.models.load_model("./Test1")
        class_names = ['Butterfly','Cat','Chicken','Cow','Dog','Elephant','Horse','Sheep','Spider','Squirrel']
        image = tf.keras.preprocessing.image.load_img(UPLOAD_FOLDER + "1." + ext, 
                                                      grayscale=False, 
                                                      color_mode='rgb', 
                                                      target_size=(100, 100),
                                                      interpolation='nearest'
                                                      )
        img = tf.keras.preprocessing.image.img_to_array(image)
        img = tf.expand_dims(img, 0)
        prediction_out = model.predict(img)
        score = tf.nn.softmax(prediction_out[0])
        animal = class_names[np.argmax(score)]
        chance = 100 * np.max(score)
        os.remove(UPLOAD_FOLDER + '1.' + ext)
        return render_template('Main.html', result = animal, chance=chance)
    
@app.route('/contrib', methods = ['GET'])
def show_contrib():
    if request.method == 'GET':
        return render_template('Contrib.html')

@app.route('/model', methods=['GET'])
def show_model():
    if request.method == 'GET':
        return render_template('Model.html')

if __name__ == '__main__':
   app.run(debug =True)