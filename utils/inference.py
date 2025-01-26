import cv2
import face_recognition
from PIL import Image
import io
import base64
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from SearchFunction import searchdb
class Inference():

    def __init__(self, image, names, feats, db_dir='D:\\Military\\Data\\MyRaw'):
        self.img = np.array(image)
        # self.img = cv2.imread(image)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.margin = 0.05
        self.min_face_h = 30
        self.min_face_w = 30
        self.names = names 
        self.feats = feats
        self.ROOT = Path(db_dir)
        self.df = pd.read_csv('D:\\Military\\Code\\final\\weights\\image_faces_flags.csv')
    def face_handler(self, boxes):
        font= cv2.FONT_HERSHEY_SIMPLEX
        persons = []
        for box in boxes:
            face = self.crop_bbox(box)
            h, w, _ = face.shape 
            max_score = 4000000000
            best_face = None
            if h < self.min_face_h and w < self.min_face_w:
                continue

            face = face_recognition.load_image_file(face)       
            # try:     
            bounding_box = [(0, w, h, 0)]
            face_embed = face_recognition.face_encodings(face, known_face_locations=bounding_box, model='large')[0]
            # except:
            #     continue   
            for person, embed in zip(self.names, self.feats):
                try:
                    score = face_recognition.face_distance([embed], face_embed)[0]
                except:
                    continue
                if score < max_score:
                    max_score = score
                    best_face = person    
            # Draw a filled rectangle for text background (black)
            text_size = cv2.getTextSize(str(best_face), font, 0.7, 2)[0]
            text_x, text_y, _, _ = box.xyxy.tolist()[0]
            text_x, text_y = int(text_x), int(text_y)
            
            if max_score < 0.6:
                print(max_score)
                persons.append(str(best_face))
                cv2.rectangle(self.img, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 0, 0), -1)    
                # Add the text (white color)
                cv2.putText(self.img, str(best_face), (text_x, text_y), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(self.img, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 0, 0), -1)    
                # Add the text (white color)
                cv2.putText(self.img, "Unkown", (text_x, text_y), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                
        return persons
    
    def flag_handler(self, boxes, classification_model):
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        flags = {}
        for box in boxes:
            flag = self.crop_bbox(box)
            h, w, _ = flag.shape
            if h < self.min_face_h and w < self.min_face_w:
                continue    
            results = classification_model(flag)
            for result in results:
                if result.probs.top1conf.item()<0.6:
                    continue
                id = result.probs.top1

                name = result.names[id]

                # Draw a filled rectangle for text background (black)
                text_size = cv2.getTextSize(name, font, 0.7, 2)[0]
                text_x, text_y, _, _ = box.xyxy.tolist()[0]
                text_x, text_y = int(text_x), int(text_y)
                cv2.rectangle(self.img, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 0, 0), -1)    
                # Add the text (white color)
                # Add the text (white color)
                cv2.putText(self.img, str(name), (text_x, text_y), font, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
            
                if name == 'Unknown':
                    continue
                if name in flags.keys():
                    flags[name] += 1
                else:
                    flags[name] = 1
        return flags   
       
    @staticmethod    
    def convert2base64(image):

        
        # Create a bytes buffer 
        buffer = io.BytesIO()
        
        # Save the image to the buffer
        image.save(buffer, format='JPEG')
        
        # Get the buffer data
        buffer.seek(0)
        image_data = buffer.read()
        
        # Convert the image data to base64
        base64_data = base64.b64encode(image_data).decode('utf-8')
        
        return base64_data
    
    @staticmethod
    def delete_files_in_directory(directory):
    # Convert the directory to a Path object
        dir_path = Path(directory)
        
        # Iterate over all files in the directory
        for item in dir_path.iterdir():
            try:
                if item.is_file():
                    item.unlink()  # Delete the file
                elif item.is_dir():
                    # Optionally, you can delete directories as well
                    item.rmdir()
            except Exception as e:
                print(f"Error deleting {item}: {e}")

    def find_largest_subset(self, query_faces, flags_dict):

        self.delete_files_in_directory(f'.\\pics\\fulls\\')
        self.delete_files_in_directory(f'.\\pics\\thumbs\\')    
        best_image_ids = searchdb(query_faces, list(flags_dict.keys()))
        if best_image_ids:
            cnt = 0
            for i, name in enumerate(best_image_ids):
                try:
                    if i>100:
                        break
                    shutil.copy(str(self.ROOT / (name + ".jpg")),
                                f'.\\pics\\fulls\\{cnt:02d}.jpg')
            
                    shutil.copy(str(self.ROOT / (name + ".jpg")),
                        f'.\\pics\\thumbs\\{cnt:02d}.jpg')
                    cnt += 1
                except:
                    continue
    def meta_data(self, detection_model_faces, detection_model_flags, classification_model ):
        dets_faces = detection_model_faces(self.img, conf=0.7, verbose=False)
        dets_flags = detection_model_flags(self.img, conf=0.01, verbose=False)
        meta = {} 
        for det in dets_flags:
            if det is None:
                continue      
            boxes = det.boxes

            flag_boxes = boxes[boxes.cls == 1]
            num_flags = len(flag_boxes)
            flag_dict = self.flag_handler(flag_boxes, classification_model)


        for det in dets_faces:
            if det is None:
                continue
            boxes = det.boxes
            face_boxes = boxes[boxes.cls == 0]
            num_faces = len(face_boxes)
            persons_list = self.face_handler(face_boxes)
            
            self.find_largest_subset(persons_list, flag_dict)


        cv2.imwrite('./Query.jpg', self.img)

        return {"Person":"Hamid"}
    
    def crop_bbox(self, box):

        x1, y1, x2, y2 = list(map(int, box.xyxy.tolist()[0]))
        expand_x = int((x2-x1) * self.margin)
        expand_y = int((y2-y1) * self.margin)
        x1 = (x1-expand_x) if (x1-expand_x)>0 else x1
        x2 = (x2+expand_x) if (x2+expand_x)<self.img.shape[1] else x2
        y1 = (y1-expand_y) if (y1-expand_y)>0 else y1
        y2 = (y2+expand_y) if (y2+expand_y)<self.img.shape[0] else y2 
        cv2.rectangle(self.img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return self.img[y1:y2, x1:x2]

