import numpy as np
from pathlib import Path
import cv2
import os
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf


from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import torch

from sqlite_handler import insert_data,create_table

from modules.KDtree import closest_cycle_person_pairs
from modules.iou import iou
from modules.face_save import face_save
from modules.face_recognizer import face_matcher


class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
cycle_person_iou = {}
bicycle_parked={}
fixed_cp_iou = np.zeros((2, 1000)).astype(int)
min_frame=3 #3
max_frame=15 #10
min_frame_for_alarm=5
iou_threshold=20
center_point_distance_theshold=3
center_point_frame_theshold=10
frame_count_threshold_for_confirm_thief=3

yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture("../videos/test/video_20230515_180551.mp4")   #test/video_20230515_180213

codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('../videos/result/results_4.mp4', codec, vid_fps, (1080,720))


def point_distance(x1,y1,x2,y2):
    return (((x2-x1)**2)+((y2-y1)**2))**0.5

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN :  
        print("x={}, y={}".format(x, y))
        
#video frame
cv2.namedWindow('FRAME')
# mouse pointer on video frame
cv2.setMouseCallback('FRAME', POINTS)


while True:
    _, img = vid.read()
    if img is None:
        for col_idx in range(fixed_cp_iou.shape[1]):
            if(fixed_cp_iou[:, col_idx][1]>0):
                print(col_idx,fixed_cp_iou[:, col_idx])
        print('Completed')
        break

    img=cv2.resize(img,(1080,720))
    # print(img.shape)
    results= yolov5_model(img)


    boxes_ = []
    scores_= []
    names_= []
    

    for index,row in results.pandas().xyxy[0].iterrows():
        x1=(int(row['xmin']))
        y1=(int(row['ymin']))
        x2=(int(row['xmax']))
        y2=(int(row['ymax']))
        cnf=float("{:.4f}".format(float(row['confidence'])))*100
        cls=str(row['name'])

        cx=(x1+x2)/2
        cy=(y1+y2)/2


        x2=(x2-x1)
        y2=(y2-y1)

        if((cls=="person" or cls=="bicycle") and cx>130 and cx<975 and cy>376):  #370
            boxes_.append([x1, y1, x2, y2])
            scores_.append(cnf)
            names_.append(cls)
            # cv2.rectangle(img, (int(x1),int(y1)), (int(x1+x2),int(y1+y2)), (255,0,0), 2)
            # cv2.putText(img, cls+"-"+str(cnf), (int(x1), int(y1-10)), 0, 0.5, (255, 255, 255), 1)

    scores_=[scores_]

    scores = np.array(scores_, dtype=np.float32)
    names = np.array(names_)

    features = encoder(img, boxes_)


    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(boxes_, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    person=[]
    bicycle=[]

    dict_of_bbox={}


    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        trackID=track.track_id

        # if trackID == 18:
        #     trackID = 12
        # if trackID == 9:
        #     trackID = 4

        center_x,center_y = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))

        dict_of_bbox[trackID]=bbox



        cv2.circle(img, (center_x,center_y), 4 , (255, 0 , 0), -1)
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,0,0), 2)
        cv2.putText(img, class_name+"-"+str(trackID), (int(bbox[0]), int(bbox[1]-10)), 0, 0.5, (255, 255, 255), 1)



        if class_name=="person":
            person.append((trackID,(center_x,center_y)))
        elif class_name=="bicycle":
            bicycle.append((trackID,(center_x,center_y)))
            if trackID not in bicycle_parked:
                bicycle_parked[trackID]=[0,0,0,center_x,center_y,0]  #[parked, frame_count, center_point_distance, cx, cy, thief_frame_count]

            bicycle_parked[trackID][2]=point_distance(bicycle_parked[trackID][3],bicycle_parked[trackID][4],int(((dict_of_bbox[trackID][0]) + (dict_of_bbox[trackID][2]))/2), int(((dict_of_bbox[trackID][1])+(dict_of_bbox[trackID][3]))/2))
            
            bicycle_parked[trackID][3]=center_x
            bicycle_parked[trackID][4]=center_y
            
            
            
    
    # print(person,bicycle)
    if (len(person)>0 and len(bicycle)>0):
        nearest_pc_pairs=closest_cycle_person_pairs(bicycle,person)

        for person_id, cycle_id in nearest_pc_pairs.items():
            
            # change3
            # cycle parked or not parked
            if (fixed_cp_iou[0][cycle_id]==person_id
                and bicycle_parked[cycle_id][0]==0):
                
                if bicycle_parked[cycle_id][2]<center_point_distance_theshold:
                    bicycle_parked[cycle_id][1]=bicycle_parked[cycle_id][1]+1
                else:
                    bicycle_parked[cycle_id][1]=0

                if (bicycle_parked[cycle_id][1]==center_point_frame_theshold):
                    bicycle_parked[cycle_id][0]=1
            # change3 end

            iou_=int(round(iou(dict_of_bbox[person_id],dict_of_bbox[cycle_id]), 3)*100)
            
            if(iou_>iou_threshold and cycle_id not in cycle_person_iou):
                cycle_person_iou.update({
                    cycle_id: {
                        person_id:[[iou_],1],
                    },
                })
                # cycle_person_iou[cycle_id][person_id]=[[iou_],1]

            elif (iou_>iou_threshold and cycle_id in cycle_person_iou):

                if(person_id in cycle_person_iou[cycle_id]):
                    cycle_person_iou[cycle_id][person_id][1]=cycle_person_iou[cycle_id][person_id][1]+1
                    if iou_>=max(cycle_person_iou[cycle_id][person_id][0]):
                        cycle_person_iou[cycle_id][person_id][0].append(iou_)

                else:
                    cycle_person_iou[cycle_id][person_id]=[[iou_],1]

                
                # fix
                cycle_dict = cycle_person_iou[cycle_id]
                person_with_max_frame = max(cycle_dict.keys(), key=lambda key: (cycle_dict[key][1], sum(cycle_dict[key][0])/len(cycle_dict[key][0])))

                if (cycle_person_iou[cycle_id][person_with_max_frame][1]>min_frame and fixed_cp_iou[0][cycle_id]==0):
                    print("\ncycle : ", cycle_id, "owner : ", person_with_max_frame)
                    fixed_cp_iou[0][cycle_id]=person_with_max_frame
                    fixed_cp_iou[1][cycle_id]=sum(cycle_person_iou[cycle_id][person_with_max_frame][0]) / len(cycle_person_iou[cycle_id][person_with_max_frame][0])
                
                # alarm
                elif (fixed_cp_iou[0][cycle_id]!=person_id 
                    and cycle_person_iou[cycle_id][person_id][1]>=min_frame_for_alarm
                    and fixed_cp_iou[0][cycle_id]!=0
                    and bicycle_parked[cycle_id][0]==1
                    and bicycle_parked[cycle_id][2]>=center_point_distance_theshold):
                    
                    bbox_img = img[int(dict_of_bbox[person_id][1]):int(dict_of_bbox[person_id][3]), int(dict_of_bbox[person_id][0]):int(dict_of_bbox[person_id][2])]
                    
                    # cv2.imwrite("./Parking_person_image/thief.jpg", bbox_img)

                    if(bicycle_parked[cycle_id][5]<frame_count_threshold_for_confirm_thief):
                        if (face_matcher(bbox_img,cycle_id)):
                            print("Alarm: Cycle : ",cycle_id," Owner : ",fixed_cp_iou[0][cycle_id], " Potential Thief : ",person_id)
                            bicycle_parked[cycle_id][5]=bicycle_parked[cycle_id][5]+1
                            # change1
                            cv2.putText(img, "Potential Thief", (int(dict_of_bbox[person_id][0]), int(dict_of_bbox[person_id][1]-20)), 0, 0.5, (0, 0, 255), 1)
                            cv2.putText(img, "Beign Theft", (int(dict_of_bbox[cycle_id][0]), int(dict_of_bbox[cycle_id][1]-20)), 0, 0.5, (0, 0, 255), 1)
                        else:
                            bicycle_parked[cycle_id][1]=1
                            bicycle_parked[cycle_id][0]=0
                            # print("face matched")
                    else:
                        cv2.putText(img, "Thief", (int(dict_of_bbox[person_id][0]), int(dict_of_bbox[person_id][1]-20)), 0, 0.5, (0, 0, 255), 1)
                        cv2.putText(img, "Beign Theft", (int(dict_of_bbox[cycle_id][0]), int(dict_of_bbox[cycle_id][1]-20)), 0, 0.5, (0, 0, 255), 1)

                        # save face of thief and cycle image
                        if(bicycle_parked[cycle_id][5]==frame_count_threshold_for_confirm_thief):

                            if not os.path.exists("./Thief_and_cycle_image/"+str(cycle_id)+"/"):
                                os.mkdir("./Thief_and_cycle_image/"+str(cycle_id)+"/")

                            cv2.imwrite("./Thief_and_cycle_image/"+str(cycle_id)+"/"+str(person_id)+".jpg", bbox_img) #thief 
                            # bbox_img_cycle = img[int(dict_of_bbox[cycle_id][1]):int(dict_of_bbox[cycle_id][3]), int(dict_of_bbox[cycle_id][0]):int(dict_of_bbox[cycle_id][2])]  #cycle
                            # cv2.imwrite("./Thief_and_cycle_image/"+str(cycle_id)+"/cycle_"+str(cycle_id)+".jpg", bbox_img_cycle)
                            print("Alarm: Cycle : ",cycle_id," Owner : ",fixed_cp_iou[0][cycle_id], " Thief : ",person_id)
                            bicycle_parked[cycle_id][5]=bicycle_parked[cycle_id][5]+1

                            insert_data((cycle_id, 9, person_id, 1))  #INSERT INTO cycle_owner_theif (Cycle_ID, Owner_ID, Thief_ID, Status)


                            

                    # else:
                    #     print("Face didn't match")
                    


                # face data save
                if (cycle_person_iou[cycle_id][person_with_max_frame][1]>min_frame and
                    cycle_person_iou[cycle_id][person_with_max_frame][1]<=max_frame and
                    fixed_cp_iou[0][cycle_id]==person_with_max_frame):
                        face_save(cycle_id,img,dict_of_bbox[fixed_cp_iou[0][cycle_id]],cycle_person_iou[cycle_id][person_with_max_frame][1])
                        # print(cycle_id,dict_of_bbox[fixed_cp_iou[0][cycle_id]],cycle_person_iou[cycle_id][person_with_max_frame][1])

                



    cv2.imshow('FRAME',img)
    out.write(img)
    

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()



# if(person_id in fixed_cp_iou[1])
