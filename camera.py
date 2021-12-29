import os
import argparse
import cv2
import sys
import numpy as np
import insightface
import fuzzyextractor
import time
import math
from insightface.app import FaceAnalysis 
from multiprocessing import Pool

if __name__=='__main__':


    # This is a super simple (but slow) example of running face recognition on live video from your webcam.
    # There's a second example that's a little more complicated but runs faster.

    # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
    # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
    # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)


    n=512
    FE=fuzzyextractor.FuzzyExtractor(n,0.075)

    keys=[]
    helpers=[]
    names=[]
    pool=Pool(1)



    parser = argparse.ArgumentParser(description='insightface app test') 
    parser.add_argument('--ctx', default=0, type=int, help='ctx id, <0 means using cpu')
    parser.add_argument('--det-size', default=640, type=int, help='detection size')
    args = parser.parse_args() 
    app = FaceAnalysis(name='antelopev2')
    app.prepare(ctx_id=args.ctx, det_size=(args.det_size,args.det_size))


    for f in os.listdir('./RPs/'):
        if f[-3:]=='key':
            name=f[0:-4]

            with open('./RPs/'+name+'.key','rb') as fi:
                R=fi.read(32)
            with open('./RPs/'+name+'.helper','r') as fi:
                P=np.array([int(x) for x in fi.readline().split()])

            keys.append(R)
            helpers.append(P)
            names.append(name) 

    proc=pool.apply_async(FE.Rep,(P,np.zeros(n)))
    name='Unknown'
    idx=0
    cnt=0

    while True:
        # Grab a single frame of video
        ret, img = video_capture.read()
        try:
            faces = app.get(img)
        except:
            continue

        rimg=img.copy()

        for face in faces[0:1]:
            img_embedding = face.normed_embedding.tolist()
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(rimg, (box[0], box[1]), (box[2], box[3]), color, 2)


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
                proc=pool.apply_async(FE.Rep,(helpers[idx],img_embedding)) 

            cv2.rectangle(rimg, (box[0], box[3] + 35), (box[2], box[3]), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(rimg, name, (box[0] , box[3] +30), font, 1.0, (255, 255, 255), 1)


        # Display the resulting image
        cv2.imshow('Video, Press Q to exit', rimg)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
