import firebase_admin
from firebase_admin import credentials, storage, db
from datetime import datetime
import uuid

from sqlite_handler import *

# Initialize Firebase Admin SDK
cred = credentials.Certificate("bicyclethief-d1cde-firebase-adminsdk-zzrcu-e2d424e307.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://bicyclethief-d1cde-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'bicyclethief-d1cde.appspot.com'
})

# Reference to the Firebase Storage bucket
bucket = storage.bucket()

# Reference to the Firebase Realtime Database
db_ref = db.reference('/cycle_thief')

def upload_image_and_data(owner_image,thief_image,cycle_id,owner_id,thief_id):
    # Upload owner-image to Firebase Storage
    owner_image_storage = str(uuid.uuid4())+'.jpg'
    owner_image_blob = bucket.blob(owner_image_storage)
    owner_image_blob.upload_from_filename(owner_image)
    owner_image_blob.make_public()
    owner_image_url = owner_image_blob.public_url
    
    # Upload thief-image to Firebase Storage
    thief_image_storage = str(uuid.uuid4())+'.jpg'
    thief_image_blob = bucket.blob(thief_image_storage)
    thief_image_blob.upload_from_filename(thief_image)
    thief_image_blob.make_public()
    thief_image_url = thief_image_blob.public_url
 
    # Get the current timestamp as a datetime object
    current_timestamp = datetime.now()

    # Convert the datetime object to a string if needed
    current_timestamp_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Store image URL and additional data in Firebase Realtime Database
    data_to_store = {
        'ownerImageUrl': owner_image_url,
        'thiefImageUrl': thief_image_url,
        'cycleId': str(cycle_id),
        'ownerId': str(owner_id),
        'thiefId': str(thief_id),
        'eventTime':current_timestamp_str
    }
    db_ref.push(data_to_store)



if __name__ == "__main__":
    print("hello")
    while True:
        result = read_data(status=1)
        id_list = []
        if len(result) == 0 :
            continue

        for r in result:
            print(r)
            Id = r[0]
            cycle_id = r[1]
            owner_id = r[2]
            thief_id = r[3]
            owner_image = './Parking_person_image/'+str(cycle_id)+'/9.jpg'
            thief_image = './Thief_and_cycle_image/'+str(cycle_id)+'/'+str(thief_id)+'.jpg'
            status=r[4]
            try:            
                upload_image_and_data(owner_image,thief_image,cycle_id,owner_id,thief_id)
                id_list.append(Id)
            except Exception as e:
                print("Exception: ",e)
        update_data(id_list,new_status=0)            
