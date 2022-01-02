from flask import Flask, request, Response, redirect, url_for
import pandas as pd
import numpy as np
from faceDetect import *
from Spotify import *


# API
app = Flask(__name__)


@app.route('/')
def index():
    return '''
        <h1>Python Face Detection</h1>
        <p>Detect your face</p>
    '''


'''
# route http POST
@app.route('/detect/face', methods=['POST'])
def imageUpload():
    # retrieve image from client
    print(request.files['image'].filename)
    print(request.files['image'].content_type)
    print(request.files['image'].mimetype_params)
    print(request.files['image'].headers)
    if request.files['image'].filename != '':

        img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        # process image
        img_processed = faceDetect(img)
        if isSuccess:
            print('success')
            # return response
            return Response(response=img_processed, status=200, mimetype="application/json")
        return Response(response=img_processed, status=402, mimetype="application/json")
    else:
        return Response(response={'No image available'}, status=400, mimetype="application/json")


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename=filename), code=301)
'''


@app.route('/<emotion>/<number>')
def spotify_music(emotion, number):
    headings = ("id", "name", "album", "artists", "year")
    print(emotion)
    df1 = music_rec(emotion)

    if int(number) > len(df1):
        df1 = df1.sample(len(df1))
    else:
        df1 = df1.sample(int(number))

    result = df1.to_json(orient='records')
    return Response(response={'{"result" :' + result + '}'}, status=200, mimetype="application/json")


@app.route('/random')
def spotify_random():
    result = [{}]
    df1 = random_music()
    result = df1.to_json(orient='records')
    result = str(result)[1:-1]
    return Response(response={'{"result" :' + result + '}'}, status=200, mimetype="application/json")


@app.route('/refresh/playlist/<emotion>', methods=['POST'])
def refresh_playlist(emotion):
    success = refreshPlaylist(emotion)
    if success:
        return Response(response='{"result": "Success"}', status=200, mimetype="application/json")
    else:
        return Response(response='{"result": "Error"}', status=400, mimetype="application/json")
