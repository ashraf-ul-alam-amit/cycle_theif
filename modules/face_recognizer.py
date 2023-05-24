import face_recognition
import os
import cv2


def face_matcher(img,cycle_id):
    source_folder = "./Parking_person_image/"+str(cycle_id)+"/"
    known_image_list=[]

    if os.path.exists(source_folder):

        for file_name in os.listdir(source_folder):
            known_image = face_recognition.load_image_file(source_folder+file_name)
            known_image_encoding = face_recognition.face_encodings(known_image)

            if len(known_image_encoding) > 0:
                known_image_list.append(known_image_encoding[0])

    # unknown_image = face_recognition.load_image_file("./Parking_person_image/thief.jpg")
    unknown_encoding = face_recognition.face_encodings(img)

    if len(unknown_encoding)>0:
        results = face_recognition.compare_faces(known_image_list, unknown_encoding[0])
        # print(results)
        return results.count(True)<results.count(False)
    else:
        return True


# image=cv2.imread("./Parking_person_image/9.jpg")
# print(face_matcher(3))

