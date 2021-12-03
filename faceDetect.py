# START Program
import os.path
import random
import cv2
import json
import uuid
import pandas as pd
from deepface import DeepFace

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID"]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
music_dist = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

isSuccess = False


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
        print('Only color image allowed')
        return json.dumps({'result': 'Color image only'})
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray)

    print("[INFO] Found {0} Faces.".format(len(faces)))

    if len(faces) == 0:
        return json.dumps({'result': 'No face detected'})
    if len(faces) > 1:
        return json.dumps({'result': 'Multiple image detected'})

    for (x, y, w, h) in faces:
        try:
            prediction = DeepFace.analyze(gray, actions=['emotion'])
            print(prediction)
            emotion = prediction['dominant_emotion']
            global isSuccess
            isSuccess = True
        except Exception as e:
            print(e)
            emotion = "Neutral"

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, emotion, (x - 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        roi_color = img[y:y + h + 50, x:x + w + 50]
        print("[INFO] Object found. Saving locally.")
        path_file = ('static/faces/%s_faces.jpg' % uuid.uuid4().hex)
        cv2.imwrite(path_file, roi_color)

    # save file
    path_file = ('static/%s.jpg' % uuid.uuid4().hex)
    status = cv2.imwrite(path_file, img)
    print("[INFO] Image %s.jpg written to filesystem: " % path_file, status)

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # cv2.imshow('Image', img)
    # cv2.waitKey(10)
    return json.dumps({'result': 'Found {0} Faces'.format(len(faces)), 'filename': path_file, 'emotion': emotion})


# Return song from emotion detected
def music_rec(emotion):
    # print('---------------- Value ------------', music_dist[show_text[0]])
    dist = 'songs/' + emotion + ".csv"
    print(dist)
    df = pd.read_csv(dist)
    df = df[['id', 'name', 'album', 'artists', 'year']]
    return df


# Randomly select music from all emotion's data
def random_music():
    emotion = random.choice(music_dist)
    dist = os.path.dirname(__file__) + '/songs/' + emotion + ".csv"
    df = pd.read_csv(dist)
    df = df[{'id', 'name', 'album', 'artists', 'year'}]
    print(df.sample())
    return df.sample()

