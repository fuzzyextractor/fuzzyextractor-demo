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


    print('generateing..')
    for f in os.listdir('./images/'):
        if f[-3:]=='jpg' or f[-3:]=='png':
            img = cv2.imread('./images/'+f)
            try:
                faces = app.get(img)
            except:
                continue

            if len(faces)!=1:
                continue
            name=f[0:-4]
            print('generateing.. '+name)
            img_embedding = faces[0].normed_embedding.tolist() 
            R,P=FE.Gen(img_embedding)  
            keys.append(R)
            helpers.append(P)
            names.append(name) 

            with open('./RPs/'+name+'.key','wb') as fo:
                fo.write(R)

            with open('./RPs/'+name+'.helper','w') as fo:
                for x in P:
                    fo.write(str(x)+' ')


            """
            with open('./RPs/'+name+'.key','rb') as fi:
                R2=fi.read(32)

            P2=[]
            with open('./RPs/'+name+'.helper','r') as fo:
                P2=[int(x) for x in fo.readline().split()]
            print(R==R2,P-P2)
            """
            
    print('done')