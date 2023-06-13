from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
# from deepface import DeepFace
from mongo_integrate import connect_mongo, add_data_to_mongo
from model import create_convolutional_model

app = Flask(__name__)

global capture,rec_frame,  switch, face, rec, out , face_out
capture=0
grey=0
neg=0
face=0
switch=1
rec=0

Expressions = {0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Neutral"}
filename = './model_weights1.hdf5'
model = create_convolutional_model(7)
model.load_weights(filename)
db = connect_mongo()
# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# #Load pretrained face detection model    
# net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
camera = cv2.VideoCapture(0)

def make_prediction(unknown):
    unknown=cv2.resize(unknown,(48,48))
    unknown=unknown/255.0
    unknown=np.array(unknown).reshape(-1,48,48,1)
    predict=np.argmax(model.predict(unknown),axis = 1)
    print(predict[0])
    return predict[0]  

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)


    
# @app.route('/detect_face',methods=['POST','GET'])
# def detect_face(img):
    

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
    
        if success:
            

            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            if(face):
                
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,1.3,5)
                ans = "unknown expression"
                for (x,y,w,h) in faces:
                    sub_face = gray[y:y+h, x:x+w]
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    # time.sleep(3)
                    res = make_prediction(sub_face)
                    ans = str(Expressions[res])
                    # add_data_to_mongo(db, ans)

                    print(f"Prediction : {ans}")
                    print("Inside Detect_Face")

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # Set the font
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    # Set the font size and color
                    font_scale = 1
                    
                    font_color = (255, 255, 255)
                    frame=cv2.putText(
                        cv2.flip(frame, 1),
                        ans,
                        (320, 80),
                        font,
                        2,
                        (0, 255, 255),
                        2,
                        cv2.LINE_4
                    ) 
                    if(capture):
                        capture=0
                        now = datetime.datetime.now()
                        p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                        cv2.imwrite(p, frame)
                    if(rec):
                        rec_frame=frame
                        frame= cv2.putText(frame,"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                        # frame=cv2.flip(frame,1)
                        # frame=frame
                    frame=cv2.flip(frame,1) 
                    
                    
                
                if ans != "unknown expression":
                        add_data_to_mongo(db, ans)
                
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        print(request.form)
        
        if request.form.get('click') == 'Capture':
            global capture
            capture=1  
            
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1 
        if  request.form.get('face') == 'Detect':
            print("face")
            global face, face_out
            face= 1
            face_out=0
                
                
            if  request.form.get('rec') == 'Start/Stop Recording':
                global rec, out
                rec= not rec
                if(rec):
                    now=datetime.datetime.now() 
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter('Recordings/vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                    #Start new thread for recording the video
                    thread = Thread(target = record, args=[out,])
                    thread.start()
                elif(rec==False):
                    out.release()
       
        
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__=="__main__":
    app.run(debug=True)
