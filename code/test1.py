import dlib
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import numpy as np

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
            cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 52, 52), 2)
            cv2.putText(frame,"stress value:{}".format(str((stress_value*100))),(10,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 66, 232), 2)
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
 
#processing real time video input to display stress 

