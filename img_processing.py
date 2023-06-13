import cv2
import numpy as np
import numpy as np
import skimage.exposure
import os
import imutils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist
import tensorflow as tf
import mediapipe as mp
import math

draw_line = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose_est = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


def create_dir(file_path) :
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def generate_Mask(img) :
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB) ##covert ke mode warna LAB
    a_channel = lab[:,:,1] ##mengambil saluran warna citra merah-hijau
    thresh = cv2.threshold(a_channel, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] #ambang batas bawah 127 ambang atas 255 
                                                                                    # menghasilkan ambang biner adaptif
    mask = cv2.bitwise_and(img, img, mask = thresh) #nilai pixel yang memiliki nilai 255 dipertahankan, yg tidak akan dihitamkan
    mask_lab= cv2.cvtColor(mask, cv2.COLOR_BGR2LAB) #convert masked ke mode warna LAB
    masked_img = cv2.cvtColor(mask_lab, cv2.COLOR_LAB2BGR) ##convert mask_lab ke warna RGB
    masked_img[thresh==0]=(0,0,0) ##background diganti warna hitam
    create_dir("masked_img") ##buat directory
    mask_path = os.path.join("masked_img","masked_img.png") #simpan foto di folder directory
    cv2.imwrite(mask_path, masked_img) 
    return masked_img

 
def find_Face(masked_img):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    faces = faceCascade.detectMultiScale(
        masked_img,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        print ("Face not found")
        return 0
    else:
        min_area = 0
        for (x, y, w, h) in faces:
            face_area = w*h
            if face_area > min_area:
                face = [x,y,w,h]
        return face
    

def get_Edged(masked_img) :
    edged = cv2.Canny(masked_img, 50, 100) ##deteksi tepian gambar
    edged = cv2.dilate(edged, None, iterations=1) ##memperluas dan mempertajam deteksi tepi (operasi dilasi)
    edged = cv2.erode(edged, None, iterations=1)
    return edged


def get_Contour(masked_img):
    edged = get_Edged(masked_img)
    c = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cari countour menggunakan edged
    c = imutils.grab_contours(c)
    (cnts, _) = contours.sort_contours(c) #mencari countour terbesar
    lst_count=[]
    for index,contour in enumerate(cnts):
        lst_count.append([index,cv2.contourArea(contour)])
    lst_count.sort(key=lambda lst_count: lst_count[1], reverse=True)
    cnts = cnts[lst_count[0][0]]
    return cnts, lst_count


def midpoint(fromP, toP):
	return ((fromP[0] + toP[0]) * 0.5, (fromP[1] + toP[1]) * 0.5)


def get_Sckeleton(img,masked_img):
    cnts, list = get_Contour(masked_img)#edged = get_Edged(masked_img)
    imgc = img.copy()
    box = cv2.minAreaRect(cnts) ##deteksi kotak terkecil yang melingkupi kontur
    x, y, w, h = cv2.boundingRect(cnts) ##get bounding box melingkupi kontur
    cv2.rectangle(imgc, (x, y), (x+w, y+h), (0, 0, 255), 3) ##return titik sudut rectangle
    box = cv2.boxPoints(box) 
    box = np.array(box, dtype="int")
    box = imutils.perspective.order_points(box) ##mengurutkan titiksudut dari kiri,kanan
    cv2.drawContours(imgc, [box.astype("int")], -1, (0, 255, 0), 3)

    for (x, y) in box:
        cv2.circle(imgc, (int(x), int(y)), 5, (0, 0, 255), -1)

    (TL, TR, BR, BL) = box
    (TLTRx, TLTRy) = midpoint(TL, TR)
    (BLBRx, BLBRy) = midpoint(BL, BR)
    (TLBLx, TLBLy) = midpoint(TL, BL)
    (TRBRx, TRBRy) = midpoint(TR, BR)

    cv2.circle(imgc, (int(TLTRx), int(TLTRy)), 5, (255, 0, 0), -1)
    cv2.circle(imgc, (int(BLBRx), int(BLBRy)), 5, (255, 0, 0), -1)
    cv2.circle(imgc, (int(TLBLx), int(TLBLy)), 5, (255, 0, 0), -1)
    cv2.circle(imgc, (int(TRBRx), int(TRBRy)), 5, (255, 0, 0), -1)

    cv2.line(imgc, (int(TLTRx), int(TLTRy)), (int(BLBRx), int(BLBRy)),(255, 0, 255), 2)
    cv2.line(imgc, (int(TLBLx), int(TLBLy)), (int(TRBRx), int(TRBRy)),(255, 0, 255), 2)
    # plt.subplot(1,3,3)
    # plt.imshow(imgc)
    return imgc, box

def get_Measurement(img, height) :
    try :
        img_ok = img.copy()
        masked_img = generate_Mask(img)
        face = find_Face(masked_img)
        contour,list_c = get_Contour(masked_img)
        x,y,w,h = cv2.boundingRect(contour)

        box = [[x,y],[x, y+h],[x+w,y+h],[x+w,y]]
        x,y,w,h = face    ##create face point 
        cv2.rectangle(img_ok,(x,y),(x+w,y+h),(0,0,250),2)
        face_points_box = [[x,y],[x, y+h],[x+w,y+h],[x+w,y]]
        face = np.array(face_points_box,dtype=int)
        (TL, BL, BR, TR) = box
        (TRx,TRy) = TR
        (BRx,BRy) = BR
        (TLx,TLy) = TL
        (BLx,BLy) = BL
        (TLTRx, TLTRy) = midpoint(TL, TR)
        (BLBRx, BLBRy) = midpoint(BL, BR)
        (TLBLx, TLBLy) = midpoint(TL,BL)
        (TRBRx, TRBRy) = midpoint(TR,BR)

    #estimasi tinggi dan lebar
        est_h = dist.euclidean ((TLTRx,TLTRy), (BLBRx, BLBRy))
        est_w = dist.euclidean ((TLBLx, TLBLy), (TRBRx, TRBRy))
        est_s = get_shoulder(img_ok)
        est_hp = get_hip(img_ok)
        est_px = pixel_per_met_md(img_ok,height)
        pixelMetric = est_h/float(height)
        final_h = est_h/pixelMetric
        if pose_detect(img) != "Invalid Image" :
            final_w = round(est_w/pixelMetric,1)
            # final_s = round(est_s*est_px,1)
            final_hp = round(est_hp*est_px,1)
        # img_final = cv2.cvtColor(masked_img,cv2.COLOR_BGR2RGB)
            return final_h, final_w, final_hp
        else :
            return None, None, None, None
    except ValueError :
        return None, None, None, None 

def pose_detect(img):
    res = pose_est.process(img)
    landmarks = res.pose_landmarks
    annotated_image = img.copy()
    draw_line.draw_landmarks(
    annotated_image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if (landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].visibility > 0.01
    and landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].visibility > 0.01):
        return landmarks
    else:
        return("Invalid Image")

    
def get_hip(image) :
    try :
        landmark_est = pose_detect(image)
        if landmark_est != 'Invalid Image' :
            h,w, _ = image.shape
            left_hip = (landmark_est.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].x,
              landmark_est.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].y)
            right_hip = (landmark_est.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x,
               landmark_est.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y)
            # x_left_hip = int(left_hip[0] * w)
            # y_left_hip = int(left_hip[1] * h)
            # x_right_hip = int(right_hip[0] * w)
            # y_right_hip = int(right_hip[1] * h)
            hip_est = math.sqrt((right_hip[0] - left_hip[0])**2 + (right_hip[1] - left_hip[1])**2)
            return hip_est
    except ValueError :
        return 'Invalid Image'

def get_shoulder(image) :
    try :
        landmark_est = pose_detect(image)
        if landmark_est != 'Invalid Image' :
            h,w, _ = image.shape
            left_shoulder = (landmark_est.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
                 landmark_est.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y)
            right_shoulder = (landmark_est.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x,
                  landmark_est.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y)
            # x_left_shoulder = int(left_shoulder[0] * w)
            # y_left_shoulder = int(left_shoulder[1] * h)
            # x_right_shoulder = int(right_shoulder[0] * w)
            # y_right_shoulder = int(right_shoulder[1] * h)
            shoulder_est = shoulder_est = math.sqrt((right_shoulder[0] - left_shoulder[0])**2 + (right_shoulder[1] - left_shoulder[1])**2) 
            return shoulder_est
    except ValueError :
        return 'Invalid Image'

def pixel_per_met_md(img,h) :
    try : 
        landmark_est= pose_detect(img)
        if landmark_est != 'Invalid Image' :
            point0 = landmark_est.landmark[mp.solutions.pose.PoseLandmark.NOSE]
            point28 = landmark_est.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
            length = ((point0.x - point28.x) ** 2 + (point0.y - point28.y) ** 2) ** 0.5
            px = h/length
            return px
    except ValueError :
        return 'Invalid Image'















