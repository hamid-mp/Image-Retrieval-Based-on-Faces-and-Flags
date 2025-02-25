import json
import cv2
import pandas as pd
from pathlib import Path
import shutil




class OpenImage2Yolo():

    def __init__(self, root):

        self.root = Path(root)
  

    @property
    def labels(self):
        (self.root / 'YOLO' / 'labels').mkdir(parents=True, exist_ok=True)
        (self.root / 'YOLO' / 'images').mkdir(parents=True, exist_ok=True)


        df = pd.read_csv((self.root / 'labels' / 'detections.csv').absolute())
        labels = self.CodeNames().set_index(0)[1].to_dict()
        filtered_by_cls = df[df['LabelName'].isin(labels.keys())]
        
        filtered_by_cls['LabelName'] = (filtered_by_cls['LabelName'].map(pd.Series(labels)))
        
        
        for index, row in filtered_by_cls.iterrows():
            name = row['ImageID']
            LabelName = row['LabelName'].replace(' ','_')
            txt_path = str(self.root / 'YOLO' / 'labels'/ (name + '.txt'))
            #img_name = name + '.jpg'
            #img_path = (self.root / 'data' / img_name).absolute()
            #img = cv2.imread(str(img_path))
            #W, H, _ = img.shape
            
            XMin, XMax, YMin, YMax = row['XMin'], row['XMax'], row['YMin'], row['YMax']

            X, Y, W, H = (self.yolo_bbox(XMin, XMax, YMin, YMax))
            print(txt_path)
            with open(txt_path, 'a') as f:
                f.write(f'{str(1)} {X} {Y} {W} {H}\n')
            self.copy_img(name)
        return filtered_by_cls
        
    def CodeNames(self, cls=['Flag']):
        df = pd.read_csv((self.root / 'metadata' / 'classes.csv').absolute(), header=None)
        
        if cls: 
            df = df.loc[df[1].isin(cls)]
        

        return df
    

    def copy_img(self, labelname):  
        img_name = labelname +  '.jpg'
        shutil.copy(str(self.root / 'data' / img_name),
                     str(self.root / 'YOLO' / 'images'/ img_name))


    @staticmethod
    def yolo_bbox(XMin, XMax, YMin, YMax):
        X_Center = ( (XMin + (XMax - XMin)/2 ))# / H
        Y_Center = ((YMin + (YMax - YMin)/2 ))# / W
        W_Bbox = ((XMax - XMin) )# / H
        H_Bbox = ((YMax - YMin) )# / W
        return (X_Center, Y_Center, W_Bbox, H_Bbox)



if __name__ == '__main__':

    label_converter = OpenImage2Yolo(r'~\fiftyone\open-images-v7\validation')

    df = label_converter.labels
