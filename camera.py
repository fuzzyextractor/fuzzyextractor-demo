import face_recognition
import cv2
import numpy as np
import fuzzyextractor
from multiprocessing import Pool
import math
import os

video_capture = cv2.VideoCapture(0)


n=128
t=0.055
FE=fuzzyextractor.FuzzyExtractor(n,t)


keys=[]
helpers=[]
names=[]
pool=Pool(1)

for f in os.listdir('.'):
    if f[-3:]=='jpg' or f[-3:]=='png':
        print(f)
        image = face_recognition.load_image_file(f)
        face_encoding = face_recognition.face_encodings(image)[0]
        R,P=FE.Gen(face_encoding)
        keys.append(R)
        helpers.append(P)
        names.append(f[0:-4])
        proc=pool.apply_async(FE.Rep,(P,face_encoding))


name = "Unknown"
idx=0
cnt=0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)

        if proc.ready():
            R_=proc.get()
            if R_==keys[idx]:
                print('Yes!')
                name=names[idx]
            else:
                cnt+=1
                if cnt>len(names)+1:
                    cnt=0
                    name = "Unknown"
            idx=(idx+1)%len(names)
            print(idx)
            proc=pool.apply_async(FE.Rep,(helpers[idx],face_encoding))

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
