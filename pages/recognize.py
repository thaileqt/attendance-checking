
import os
from datetime import datetime, timedelta
import pandas
import time
import pickle
import cv2
import streamlit as st
from datetime import datetime


detecting = True
new_attendance = False
checked = []
with open('model/knn.pkl', 'rb') as f:
    knn = pickle.load(f)

def gen_frames():
    global detecting, new_attendance, checked
    video = cv2.VideoCapture(0)
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    check_time = 5
    current = None
    checking = False
    while detecting:
        success, frame = video.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            checking = False
            current = None
            new_attendance = False
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(crop_img, (48, 48)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            if output[0] is not None and output[0] not in checked:
                if not checking:
                    current = output[0]
                    checking = True
                    start = time.time()
                else:
                    if current == output[0]:
                        if time.time() - start > check_time:
                            date = datetime.now()
                            filename = f'attendance_{date.strftime("%d-%m-%Y")}.csv'
                            with open(f'attendance/{filename}', 'a') as f:
                                f.write(f'{output[0]},{date.strftime("%d-%m-%Y %H:%M:%S")}\n')
                            checked.append(current)
                            checking = False
                            new_attendance = True

            if checking:
                cv2.putText(frame, str(int(check_time - (time.time() - start)) + 1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, output[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield frame

    video.release()
    cv2.destroyAllWindows()



def read_attendance_data(filename) -> pandas.DataFrame:
    if filename not in os.listdir('attendance'):
        with open(f'attendance/{filename}', 'w') as file:
            file.write('name,datetime\n')
    return pandas.read_csv(f'attendance/{filename}')

def main():
    global new_attendance
    st.title("Attendance System")

    image_placeholder = st.empty()
    attendace_placeholder = st.empty()

    date = datetime.now()
    filename = f'attendance_{date.strftime("%d-%m-%Y")}.csv'

    frame_generator = gen_frames()
    attendace_placeholder.dataframe(read_attendance_data(filename))
    while True:
        if new_attendance:
            new_attendance = False
            df = read_attendance_data(filename)
            attendace_placeholder.dataframe(df)

        # Get the next frame
        frame_bytes = next(frame_generator)
        image_placeholder.image(frame_bytes, channels="BGR", use_column_width=True)

if __name__ == '__main__':
    main()
