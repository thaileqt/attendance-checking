
import os
from time import time
import cv2
import pickle
import numpy as np
import subprocess
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier


capturing = False

def gen_frames(name, n_images_to_collect=50):
    global capturing
    video = cv2.VideoCapture(0)
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces_data = []
    while capturing:
        success, frame = video.read()
        if not success:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (48, 48))
            if len(faces_data) <= n_images_to_collect:
                faces_data.append(resized_img)
            cv2.putText(frame, f'{str(len(faces_data))}/{n_images_to_collect}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        if len(faces_data) >= n_images_to_collect:
            capturing = False
            print('Done collecting images')

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield frame, faces_data

    video.release()
    # cv2.destroyAllWindows()

def save_train_data(faces_data, name, n_images_to_collect):
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape((faces_data.shape[0], -1))

    # Save the labels
    if 'model' not in os.listdir():
        os.mkdir('model')
    if 'names.pkl' not in os.listdir('model'):
        names = [name] * n_images_to_collect
        with open('model/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('model/names.pkl', 'rb') as f:
            names = pickle.load(f)
            names = names + [name] * n_images_to_collect
        with open('model/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Save the faces data (numpy array)
    if 'faces_data.pkl' not in os.listdir('model'):
        with open('model/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('model/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
            faces = np.append(faces, faces_data, axis=0)
        with open('model/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)


def train():
    with open('model/names.pkl', 'rb') as f:
        LABELS = pickle.load(f)
    with open('model/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    print(len(LABELS), FACES.shape)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)
    with open('model/knn.pkl', 'wb') as f:
        pickle.dump(knn, f)


def main():
    global capturing
    n_images_to_collect = 50
    st.title("Add Face")
    status_placeholder = st.empty()
    name_placeholder = st.empty()
    button_placeholder = st.empty()
    image_placeholder = st.empty()
    name = name_placeholder.text_input('Enter your name', '')
    save_button = button_placeholder.button("Save")
    #
    if save_button:
        name_placeholder.empty()
        button_placeholder.empty()
        capturing = True

        frame_generator = gen_frames(name, n_images_to_collect)
        while capturing:
            # Get the next frame
            frame_bytes, faces_data = next(frame_generator)
            image_placeholder.image(frame_bytes, channels="BGR", use_column_width=True)
        image_placeholder.empty()
        status_placeholder.write('Saving data...')
        save_train_data(faces_data, name, n_images_to_collect)
        status_placeholder.write('Training model...')
        train()
        status_placeholder.write('Done training model')
        # list number of unique name in names.pkl
        with open('model/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names = list(set(names))
        status_placeholder.write(f'Number of trained person: {len(names)}')


if __name__ == '__main__':
    main()
    # train()
