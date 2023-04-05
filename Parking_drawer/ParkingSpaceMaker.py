import cv2
import pickle
import numpy as np
 

vid = cv2.VideoCapture("../data/video/cars.mp4")


points_list=[]

try:
    with open('CycleParkingYY', 'rb') as f:
        pair_yy = pickle.load(f)
except:
    pair_yy = (0,1000)

try:
    with open('CycleParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []


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

 
def mouseClick(events, x, y, flags, params):

    if events == cv2.EVENT_LBUTTONDOWN:
        points_list.append((int(x), int(y)))
        print(points_list)
        print(posList)
        if(len(points_list)==4):
            shortest_y = min(points_list, key=lambda t: t[1])
            largest_y = max(points_list, key=lambda t: t[1])

            posList.append(points_list[:])

            yh=0
            yl=0                 

            # Update the y value of point1 with the shortest y value
            if shortest_y[1] < pair_yy[1]:
                yl = shortest_y[1]
            else:
                yl = pair_yy[1]
            
            if largest_y[1] > pair_yy[0]:
                yh = largest_y[1]
            else:
                yh=pair_yy[0]

            points_list.clear()
            print((yh,yl))
            with open('CycleParkingYY', 'wb') as f:
                pickle.dump((yh,yl), f)

            with open('CycleParkPos', 'wb') as f:
                pickle.dump(posList, f)
            return

    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            if point_inside_quadrilateral((x,y), pos):
                posList.pop(i)
                with open('CycleParkPos', 'wb') as f:
                    pickle.dump(posList, f)
                return
 
    
 
 
while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
    # img = cv2.imread('cycle_parking.jpg')
    
    # img=cv2.resize(img,(800,600))
    # print(img.shape)

    for area_1 in posList:
        cv2.polylines(img,[np.array(area_1,np.int32)],True,(0,255,0),2)
 
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    cv2.waitKey(0)