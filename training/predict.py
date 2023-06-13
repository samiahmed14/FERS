import cv2
import numpy as np
from model import create_convolutional_model
from mongo_integrate import add_data_to_mongo, connect_mongo

def make_prediction(unknown):
    unknown=cv2.resize(unknown,(48,48))
    unknown=unknown/255.0
    unknown=np.array(unknown).reshape(-1,48,48,1)
    predict=np.argmax(model.predict(unknown),axis = 1)
    return predict[0]  

def face_in_video():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, img=cap.read()  
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        ans = "unknown expression"
        for (x,y,w,h) in faces:
            sub_face = gray[y:y+h, x:x+w]
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            res = make_prediction(sub_face)
            ans = str(Expressions[res])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                img,
                ans,
                (50, 50),
                font,
                3,
                (0, 255, 255),
                2,
                cv2.LINE_4
            )
        cv2.imshow('img',img)
        if ans != "unknown expression":
            add_data_to_mongo(db, ans)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()  

if __name__=='__main__':
    Expressions = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}
    filename = 'model_weights.hdf5'
    model = create_convolutional_model(7)
    model.load_weights(filename)
    db = connect_mongo()
    face_in_video()