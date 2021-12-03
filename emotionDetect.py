# START Program
import os.path
import cv2
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import json
import uuid
from flask import Flask, request, Response, redirect, url_for
import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('Final_model.h5')


#model prediction
def model_analyze(image):

    resp_obj = {}

    final_image = cv2.resize(image, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)

    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    emotion_predictions = model.predict(final_image)[0, :]
    sum_of_predictions = emotion_predictions.sum()

    resp_obj["emotion"] = {}

    for i in range(0, len(emotion_labels)):
        emotion_label = emotion_labels[i]
        emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
        resp_obj["emotion"][emotion_label] = emotion_prediction

    resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]
    return resp_obj

# check grayscale
def is_gray_scale(img):
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False


# Function to detect face from image
def faceDetect(img):
    # draw rectangle across the face
    if is_gray_scale(img):
        print('Please input color image')
        return json.dumps({'result': 'Color image only'})
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)
    emotion = ''

    print("[INFO] Found {0} Faces.".format(len(faces)))

    if len(faces) == 0:
        return json.dumps({'result': 'No face detected'})
    for (x, y, w, h) in faces:
        try:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_color = img[y:y + h + 50, x:x + w + 50]

            prediction = model_analyze(img)
            print(prediction)
            emotion = prediction['dominant_emotion']

            cv2.putText(img, emotion, (x - 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            print("[INFO] Object found. Saving locally.")
            path_file = ('static/faces/%s_faces.jpg' % uuid.uuid4().hex)
            cv2.imwrite(path_file, roi_color)
        except Exception as e:
            print(e)
            emotion = "Neutral"


    # save file
    path_file = ('static/%s.jpg' % uuid.uuid4().hex)
    status = cv2.imwrite(path_file, img)
    print("[INFO] Image %s.jpg written to filesystem: " % path_file, status)

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # cv2.imshow('Image', img)
    # cv2.waitKey(10)

    return json.dumps({'result': 'Found {0} Faces'.format(len(faces)), 'filename': path_file, 'emotion': emotion})


# API
app = Flask(__name__)


@app.route('/')
def index():
    return '''
        <h1>Emotion Detection</h1>
        <p>Detect using OpenCV in python</p>
    '''


# route http POST
@app.route('/detect/face', methods=['POST'])
def imageUpload():
    # retrieve image from client
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    cv2.imshow('image', img)
    cv2.waitKey()
    # process image
    img_processed = faceDetect(img)
    # return response
    return Response(response=img_processed, status=200, mimetype="application/json")


@app.route('/detect/face/video', methods=['POST'])
def videoUpload():
    # retrieve image from client
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.units), cv2.IMREAD_UNCHANGED)
    cv2.imshow('image', img)
    # process image
    img_processed = faceDetect(img)
    # return response
    return Response(response=img_processed, status=200, mimetype="application/json")


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=filename), code=301)


# start server
app.run(host="0.0.0.0", port=5000)
