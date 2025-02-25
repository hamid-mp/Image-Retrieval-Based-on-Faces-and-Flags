# This python script convert VoTT JSON annotation format to YOLO Darknet TXT annotation format
# Written by Oussama Ben Zayed, a Software Engineering Student based in TUNISA 


import json
import os
VoTT_JSON_annotation_folder_path = "/media/user/60E2546EE2544B04/Military Project Data/VOTT/output/"
YOLO_Darknet_TXT_folder_path =  "/media/user/60E2546EE2544B04/Military Project Data/VOTT/output_yolo/"
jsonFiles = os.listdir(VoTT_JSON_annotation_folder_path) # enter the path of the folder where there there are the VoTT JSON annotation files

for jsonFile in jsonFiles:
    try:
        # reading json from file
        f = open(VoTT_JSON_annotation_folder_path + jsonFile)
        data = json.load(f)
        l = len(data['asset']['name'])
        yoloFileName = data['asset']['name'][:l - 3] + "txt" # extract the image name and change the extension from .jpg or .png
                                                            # to .txt (see YOLO Darknet annotaion format)
        
        width = data['asset']['size']['width'] # image width
        height = data['asset']['size']['height'] # image height
    except:
        continue
    for i in range(len(data['regions'])): # loop through all the classes of the labeled image

        x1 = data['regions'][i]['points'][0]['x']
        y1 = data['regions'][i]['points'][0]['y']
        x2 = data['regions'][i]['points'][1]['x']
        y2 = data['regions'][i]['points'][1]['y']
        x3 = data['regions'][i]['points'][2]['x']
        y3 = data['regions'][i]['points'][2]['y']
        x4 = data['regions'][i]['points'][3]['x']
        y4 = data['regions'][i]['points'][3]['y']
        
        x_center = ((x1 + x2) / 2) / width
        y_center = ((y1 + y3) / 2) / height
        w = (x2 - x1) / width
        h = (y3 - y2) / height
        if data['regions'][i]['tags'][0] == 'Flag':
            classID = '1'
        else:
            classID='0'
        
        yoloAnnotation = classID + " " + str(x_center) + " " + str(y_center) + " " + str(w) + " " + str(h) +"\n"
        yoloFile = open(YOLO_Darknet_TXT_folder_path + yoloFileName, 'a') # enter the path of the folder where there YOLO Darknet TXT annotation files will be created
        yoloFile.write(yoloAnnotation)
    yoloFile.close()
    f.close()