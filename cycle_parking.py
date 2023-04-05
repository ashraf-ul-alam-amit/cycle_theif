import numpy as np
from pathlib import Path
import cv2
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf


from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import torch

from modules.KDtree import closest_red_black_pairs
from modules.iou import iou


class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
cycle_person_iou = np.zeros((3, 1000))
fixed_cp_iou = np.zeros((2, 1000))
min_frame=5

yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# try:
#     with open('./Parking_drawer/CycleParkPos', 'rb') as f:
#         posList = pickle.load(f)
# except:
#     posList = []

# try:
#     with open('CycleParkingYY', 'rb') as f:
#         pair_yy = pickle.load(f)
# except:
#     pair_yy = (440,220)

# parking_slot=[]
# parking_slot = [[] for _ in range(len(posList))]

# def point_inside_quadrilateral(point,vertices):
#     x, y = point
#     side1 = (vertices[1][0]-vertices[0][0])*(y-vertices[0][1]) - (x-vertices[0][0])*(vertices[1][1]-vertices[0][1])
#     side2 = (vertices[2][0]-vertices[1][0])*(y-vertices[1][1]) - (x-vertices[1][0])*(vertices[2][1]-vertices[1][1])
#     side3 = (vertices[3][0]-vertices[2][0])*(y-vertices[2][1]) - (x-vertices[2][0])*(vertices[3][1]-vertices[2][1])
#     side4 = (vertices[0][0]-vertices[3][0])*(y-vertices[3][1]) - (x-vertices[3][0])*(vertices[0][1]-vertices[3][1])
#     if (side1 >= 0 and side2 >= 0 and side3 >= 0 and side4 >= 0) or (side1 <= 0 and side2 <= 0 and side3 <= 0 and side4 <= 0):
#         return True
#     else:
#         return False

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture("./data/video/bag.mp4")

# codec = cv2.VideoWriter_fourcc(*'XVID')
# vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
# vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

while True:
    _, img = vid.read()
    if img is None:
        for col_idx in range(fixed_cp_iou.shape[1]):
            if(fixed_cp_iou[:, col_idx][1]>0):
                print(col_idx,fixed_cp_iou[:, col_idx])
        print('Completed')
        break
    
    # print(len(posList))
    # for parking_id, area_1 in enumerate(posList):
    #     cv2.polylines(img,[np.array(area_1,np.int32)],True,(0,255,0),2)
    #     cv2.putText(img, str(parking_id), (area_1[0][0], area_1[0][1]-10), 0, 0.75, (255, 255, 255), 2)


    results= yolov5_model(img)


    boxes_ = []
    scores_= []
    names_= []

    for index,row in results.pandas().xyxy[0].iterrows():
        x1=(int(row['xmin']))
        y1=(int(row['ymin']))
        x2=(int(row['xmax']))
        y2=(int(row['ymax']))
        cnf=float("{:.7f}".format(float(row['confidence'])))
        cls=str(row['name'])

        # print((x1,y1))

        x2=(x2-x1)
        y2=(y2-y1)

        if(cls=="person" or cls=="suitcase"):
            boxes_.append([x1, y1, x2, y2])
            scores_.append(cnf)
            names_.append(cls)

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
        center_x,center_y = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        # print((bbox[2],bbox[3]))

        dict_of_bbox[trackID]=bbox



        # cv2.circle(img, (center_x,center_y), 4 , (255, 0 , 0), -1)
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,0,0), 2)
        cv2.putText(img, class_name+"-"+str(trackID), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)



        if class_name=="person":
            person.append((trackID,(center_x,center_y)))
        elif class_name=="suitcase":
            bicycle.append((trackID,(center_x,center_y)))
        
    # print(person,bicycle)
    if len(person)>0 and len(bicycle)>0:
        nearest_pc_pairs=closest_red_black_pairs(bicycle,person)

        for person_id, cycle_id in nearest_pc_pairs.items():
            iou_=round(iou(dict_of_bbox[person_id],dict_of_bbox[cycle_id]), 3)
            
            if(iou_>0.100 and cycle_person_iou[0][cycle_id]==0):
                cycle_person_iou[0][cycle_id]=person_id
                cycle_person_iou[1][cycle_id]=iou_
                cycle_person_iou[2][cycle_id]=cycle_person_iou[2][cycle_id]+1

            elif(iou_>0.100 and cycle_person_iou[0][cycle_id]==person_id and iou_>=cycle_person_iou[1][cycle_id]):
                # cycle_person_iou[0][cycle_id]=person_id
                cycle_person_iou[1][cycle_id]=iou_
                cycle_person_iou[2][cycle_id]=cycle_person_iou[2][cycle_id]+1

            elif(iou_>0.100 and cycle_person_iou[0][cycle_id]!=person_id and iou_>cycle_person_iou[1][cycle_id]):
                cycle_person_iou[0][cycle_id]=person_id
                cycle_person_iou[1][cycle_id]=iou_
                cycle_person_iou[2][cycle_id]=1    # person changed so frame count reset


            if (cycle_person_iou[2][cycle_id]>=min_frame and fixed_cp_iou[0][cycle_id]==0):
                fixed_cp_iou[0][cycle_id]=cycle_person_iou[0][cycle_id]
                fixed_cp_iou[1][cycle_id]=cycle_person_iou[1][cycle_id]
            elif (cycle_person_iou[2][cycle_id]>=min_frame and fixed_cp_iou[0][cycle_id]!=cycle_person_iou[0][cycle_id]):
                print("alarm ",cycle_id,fixed_cp_iou[0][cycle_id])



    cv2.imshow('frame',img)
    # out.write(img)
    

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
# out.release()
cv2.destroyAllWindows()