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

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]


yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

try:
    with open('./Parking_drawer/CycleParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

try:
    with open('CycleParkingYY', 'rb') as f:
        pair_yy = pickle.load(f)
except:
    pair_yy = (440,220)

parking_slot=[]
parking_slot = [[] for _ in range(len(posList))]

def point_inside_quadrilateral(point,vertices):
    x, y = point
    side1 = (vertices[1][0]-vertices[0][0])*(y-vertices[0][1]) - (x-vertices[0][0])*(vertices[1][1]-vertices[0][1])
    side2 = (vertices[2][0]-vertices[1][0])*(y-vertices[1][1]) - (x-vertices[1][0])*(vertices[2][1]-vertices[1][1])
    side3 = (vertices[3][0]-vertices[2][0])*(y-vertices[2][1]) - (x-vertices[2][0])*(vertices[3][1]-vertices[2][1])
    side4 = (vertices[0][0]-vertices[3][0])*(y-vertices[3][1]) - (x-vertices[3][0])*(vertices[0][1]-vertices[3][1])
    if (side1 >= 0 and side2 >= 0 and side3 >= 0 and side4 >= 0) or (side1 <= 0 and side2 <= 0 and side3 <= 0 and side4 <= 0):
        return True
    else:
        return False

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture("./data/video/cars.mp4")

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/results.avi', codec, vid_fps, (vid_width, vid_height))

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
    
    # print(len(posList))
    for parking_id, area_1 in enumerate(posList):
        cv2.polylines(img,[np.array(area_1,np.int32)],True,(0,255,0),2)
        cv2.putText(img, str(parking_id), (area_1[0][0], area_1[0][1]-10), 0, 0.75, (255, 255, 255), 2)


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

        x2=(x2-x1)
        y2=(y2-y1)

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


    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        trackID=track.track_id

        center_x,center_y = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))

        cv2.circle(img, (center_x,center_y), 4 , (255, 0 , 0), -1)
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,0,0), 2)
        cv2.putText(img, class_name+"-"+str(trackID), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)

        if class_name=="bicycle" and center_y <= pair_yy[0] and center_y >= pair_yy[1]:
            for parking_id, area_1 in enumerate(posList):
                # result=cv2.pointPolygonTest(np.array(area_1,np.int32),(((bbox[2]) - (bbox[0])),((bbox[3]) - (bbox[1]))),False)

                # if track.track_id not in parking_slot[parking_id] and result>0 :
                #     parking_slot[parking_id].append(track.track_id)

                if trackID not in parking_slot[parking_id] and point_inside_quadrilateral((center_x,center_y),area_1):
                    parking_slot[parking_id].append(trackID)

                    bbox_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    bbox_path = Path(f'./Parking_person_image/{parking_id}_{trackID}.jpg')
                    cv2.imwrite(str(bbox_path), bbox_img)
                    cv2.imshow('Bounding Box', bbox_img) 
                    # cv2.waitKey(0)

                elif track.track_id in parking_slot[parking_id] and not point_inside_quadrilateral((center_x,center_y),area_1):
                    parking_slot[parking_id].pop(parking_slot[parking_id].index(trackID))

        print(parking_slot)



    cv2.imshow('frame',img)
    out.write(img)

    if cv2.waitKey(0) == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()