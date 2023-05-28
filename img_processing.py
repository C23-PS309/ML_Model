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
    return cnts


def midpoint(fromP, toP):
	return ((fromP[0] + toP[0]) * 0.5, (fromP[1] + toP[1]) * 0.5)


def get_Sckeleton(img,masked_img):
    cnts = get_Contour(masked_img)#edged = get_Edged(masked_img)
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
    return box

def get_Measurement(img, height) :
    img_ok = img.copy()
    masked_img = generate_Mask(img)
    face = find_Face(masked_img)
    contour = get_Contour(masked_img)
    x,y,w,h = cv2.boundingRect(contour)
    box = [[x,y],[x, y+h],[x+w,y+h],[x+w,y]]
    # new_box = np.array(box,dtype=int)
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
    # left_center_x = (dist.euclidean(TL[0],BL[0]))/2
    left_center_x = (TL[0] + BL[0])/2
    right_center_x = (TR[0] + BR[0])/2 
    # blue = (42, 90, 118)
    blue = (20, 40, 60)
    green = (44, 84, 62)
    white = (255,255,255)

    cv2.circle(img_ok,(int(TLTRx), int(TLTRy)), 5, blue, -1)
    cv2.circle(img_ok, (int(BLBRx), int(BLBRy)), 5, blue, -1)
    cv2.line(img_ok, (int(TLTRx), int(TLTRy)), (int(TLx-10), int(TLy)),blue, 1)
    cv2.line(img_ok, (int(BLBRx), int(BLBRy)), (int(BLx-10), int(BLy)),blue, 1)  
    cv2.line(img_ok, (int(TLx-10), int(TLy)), (int(BLx-10), int(BLy)),blue, 1)  

    #estimasi tinggi dan lebar
    est_h = dist.euclidean ((TLTRx,TLTRy), (BLBRx, BLBRy))
    est_w = dist.euclidean ((TLBLx, TLBLy), (TRBRx, TRBRy))
    # est_chest = dist.euclidean(TRx,TLx)
    # est_hip = dist.euclidean(BRx,BLx)

    pixelMetric = est_h/float(height)
    #size object
    final_h = est_h/pixelMetric
    final_w = est_w/pixelMetric
    final_chest = abs(TRx - TLx)/pixelMetric
    final_hip = abs(BRx - BLx)/pixelMetric
    final_waist = abs(right_center_x - left_center_x)/pixelMetric

    cv2.putText(masked_img,"{:.1f}".format(final_h),(int(TLx-80),int(TLy+200)),cv2.FONT_HERSHEY_SIMPLEX,0.7,white,2)
    # cv2.putText(masked_img,"{:.1f}".format(final_w),(int(TRx),int(TRy)),cv2.FONT_HERSHEY_SIMPLEX,0.7,white,2)
    cv2.putText(masked_img,"cm",(int(TLx-55),int(TLy+220)),cv2.FONT_HERSHEY_SIMPLEX,0.7,white,2)
    # cv2.putText(masked_img,"cm",(int(TRx),int(TRy)),cv2.FONT_HERSHEY_SIMPLEX,0.7,white,2)

    img_final = cv2.cvtColor(masked_img,cv2.COLOR_BGR2RGB)
    # plt.imshow(img_final)
    # return img_final, pixelMetric
    return final_h,final_chest,final_hip,final_waist

    # return final_h, final_w













