import dlib
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import cv2
from imutils import face_utils
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model", custom_objects=None,
    compile=False)


class VideoCamera(object):
    def __init__(cap):
        #real time video capture
        cap.video = cv2.VideoCapture(0)
    def __del__(cap):
        cap.video.release()
        
    def get_frame(cap):
    #    while(True):    
        ret,frame = cap.video.read()
        frame = cv2.flip(frame,1)
        frame = imutils.resize(frame, width=500,height=500)
        #gettting points of eye from the facial landmark
        (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
        #getting eye points from facial landmarks
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # getting lip points from facial landmarks
        (l_lower, l_upper) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        #preprocessing the image
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        detections = detector(gray,0)
        for detection in detections:
            emotion= emotion_finder(detection,gray)
            cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            shape = predictor(frame,detection)
            shape = face_utils.shape_to_np(shape)
            
            # detect faces in the frame and determine if they are wearing a
        	# face mask or not
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            
            # loop over the detected face locations and their corresponding
        	# locations
            for (box, pred) in zip(locs, preds):
        		# unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
    
        		# determine the class label and color we'll use to draw
        		# the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
        		# include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            if label=="No Mask":
               
                leyebrow = shape[lBegin:lEnd]
                reyebrow = shape[rBegin:rEnd]
                openmouth = shape[l_lower:l_upper]
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                # figuring out convex shape 
                reyebrowhull = cv2.convexHull(reyebrow)
                leyebrowhull = cv2.convexHull(leyebrow)
                openmouthhull = cv2.convexHull(openmouth) 
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                
                cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [openmouthhull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                # Measuring lip distance and eye distance
                lipdist = lpdist(openmouthhull[-1],openmouthhull[0])
                eyebrowdist = ebdist(leyebrow[-1],reyebrow[0])
                eyedist=edist(leftEye[-1], rightEye[0])
                stress_value,stress_label = normalize_values(points,eyebrowdist, points_lip, lipdist, points_eye, eyedist)
    
            	  
                #displaying stress levels and value 
                # display the label and bounding box rectangle on the output frame
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 52, 52), 2)
                cv2.putText(frame,"stress value:{:.2f}%".format((stress_value*100)),(10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 66, 232), 2)
                cv2.putText(frame,"Stress level:{}".format((stress_label)),(10,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (35, 189, 25), 2)
            
            else:
                
                leyebrow = shape[lBegin:lEnd]
                reyebrow = shape[rBegin:rEnd]
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                
                # figuring out convex shape 
                reyebrowhull = cv2.convexHull(reyebrow)
                leyebrowhull = cv2.convexHull(leyebrow)
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                
                cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                
                # Measuring eyebrow distance and eye distance
                eyebrowdist = ebdist(leyebrow[-1],reyebrow[0])
                eyedist=edist(leftEye[-1], rightEye[0])
                stress_value,stress_label = normalize_values_mask(points,eyebrowdist, points_eye, eyedist)
    
            	  
                #displaying stress levels and value 
                # display the label and bounding box rectangle on the output frame
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 52, 52), 2)
                cv2.putText(frame,"stress value:{:.2f}%".format((stress_value*100)),(10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 66, 232), 2)
                cv2.putText(frame,"Stress level:{}".format((stress_label)),(10,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (35, 189, 25), 2)
            #cv2.imshow("Frame", frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


global points, points_lip, emotion_classifier, detector, predictor

#importing frontal facial landmark detector        
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#loading the trained model
emotion_classifier = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
    
points=[]; points_lip=[]; points_eye=[]
    
#calculating eye distance in terms of the facial landmark
def ebdist(leye,reye):
    eyedist = dist.euclidean(leye,reye)
    points.append(int(eyedist))
    return eyedist

#calculating eye distance in terms of the facial landmark

def edist(leye,reye):
    eyedist = dist.euclidean(leye,reye)
    points_eye.append(int(eyedist))
    return eyedist

#calculating lip dostance using facial landmark
def lpdist(l_lower,l_upper):
    lipdist = dist.euclidean(l_lower, l_upper)
    points_lip.append(int(lipdist))
    return lipdist

# Mask Detection Function


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

#finding stressed or not using the emotions 
def emotion_finder(faces,frame):
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ['scared','sad','angry']:
        label = 'Stressed'
    else:
        label = 'Not Stressed'
    return label

#calculating stress value using the distances
def normalize_values(points,disp,points_lip,dis_lip, points_eye, eyedist):
    
    normalized_value_lip = abs(dis_lip - np.min(points_lip))/abs(np.max(points_lip) - np.min(points_lip))
    normalized_value_eyebrow =abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    normalized_value_eye =abs(eyedist - np.min(points_eye))/abs(np.max(points_eye) - np.min(points_eye))
    normalized_value1 =( normalized_value_eye + normalized_value_eyebrow)/2
    normalized_value2=(normalized_value1 + normalized_value_lip)/2
    stress_value = (np.exp(-(normalized_value2)))
    if stress_value>=0.65:
        stress_label="High Stress"
    else:
        stress_label="Low Stress"
    return stress_value,stress_label

def normalize_values_mask(points,disp, points_eye, eyedist):
    
    normalized_value_eyebrow =abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    normalized_value_eye =abs(eyedist - np.min(points_eye))/abs(np.max(points_eye) - np.min(points_eye))
    normalized_value =( normalized_value_eye + normalized_value_eyebrow)/2
    stress_value = (np.exp(-(normalized_value)))
    if stress_value>=0.65:
        stress_label="High Stress"
    else:
        stress_label="Low Stress"
    return stress_value,stress_label
 
#processing real time video input to display stress 

