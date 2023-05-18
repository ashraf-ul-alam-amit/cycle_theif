import os
import cv2


def face_save(dir,img,bbox,img_id):
    source_folder = r"./Parking_person_image/"

    if not os.path.exists("./Parking_person_image/"+str(dir)+"/"):
        os.mkdir("./Parking_person_image/"+str(dir)+"/")

    bbox_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    cv2.imwrite("./Parking_person_image/"+str(dir)+"/"+str(img_id)+".jpg", bbox_img)