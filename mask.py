import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import time
import cv2
import os
from keras.utils import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import imutils


class mask_detector:

    def __init__(self, frame, facenet, masknet):
        
        self.frame = frame
        self.facenet = facenet
        self.masknet = masknet

    def detect_predict_mask(self):

        (h,w) = self.frame.shape[:2] #dimensiones del marco
        blob = cv2.dnn.blobFromImage(self.frame, 1.0, (224,224), (104.0, 177.0, 123.0))#procesamiento de las imagenes
        #pasamos el blob a la red para detectar las caras

        self.facenet.setInput(blob) 
        detections = self.facenet.forward()
    

        faces = []
        locs = []
        preds = []

        for i in range(0,detections.shape[2]):
            
            conf = detections[0,0,i,2]

            if conf > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = self.frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        return (locs, preds)

prototxt = r"/home/airamariza/Cosas/proyectos/mask_pro/caffe_model_for_dace_detection/deploy.prototxt.txt"
caffe = r"/home/airamariza/Cosas/proyectos/mask_pro/caffe_model_for_dace_detection/res10_300x300_ssd_iter_140000.caffemodel" 
faceNet = cv2.dnn.readNetFromCaffe(prototxt, caffe)
maskNet = load_model("/home/airamariza/Cosas/proyectos/mask_pro/mask_detector_model.h5")

vs = VideoStream(src=0).start()


while True:
	#inicializamos la variable frame y le asignamos que tenga una anchura de 1200 pixeles
	frame = vs.read()
	frame = imutils.resize(frame, width= 1200)

	#llamamos a la clase para detectar las caras en el frame y decir si llevan o no mascarilla 
	(locs, preds) = mask_detector(frame, faceNet, maskNet).detect_predict_mask()

    #iteramos sobre las localizaciones y las caras detectadas
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred, (1-pred)

        #Determinamos a que clase pertenece y que color le corresponde a cada clase
		label = "Lleva mascarilla" if mask > withoutMask else "No lleva mascarilla"
		color = (0, 255, 0) if label == "Lleva mascarilla" else (0,0,255)

		#incluimos el porcentaje en la predicción
		label = f"{label} {int(max(mask, withoutMask) *100)}% "
		
        #mostramos la predicción de a que clase pertenece, el rectangulo para
        #que se vea en el frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	#para romper el bucle presionamos la tecla "q"
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()




