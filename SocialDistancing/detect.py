import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import load_classifier
from scipy.spatial import distance as dist

model = attempt_load('last.pt')  # load FP32 model
stride = int(model.stride.max())  # model stride()--By how may pixels it has to move
imgsz = check_img_size(640, s=stride)  # check image size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

vid = cv2.VideoCapture('PedestriansWalking.mp4')   
while(True):
    ret, img = vid.read()
    if img is None:
        break
    img2 = img
    img1 = img
    img = letterbox(img)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416, #HWC #CWH, OpenCV,Pillow
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]
    pred = non_max_suppression(pred)
    
    # Rescale boxes from img_size to im0 size
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img1.shape).round()
            coords=[]
            cords=[]
            for *xyxy, conf, cls in reversed(det):
                coords += [(int(xyxy[0]), int(xyxy[3]))] 
                coords += [(int(xyxy[2]), int(xyxy[3]))]
                cords+=[(int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3]))]
            distances=[]
            unsafePos=set()
            cenlen=len(cords)
            D = dist.cdist(coords, coords, metric="euclidean")
            for k in range(cenlen*2):
                for j in range(cenlen*2):
                    if D[k][j]<150 and k != j :
                        if k % 2 == 0 : 
                            if k+1 != j:
                                unsafePos.add((int(cords[k//2][0]),int(cords[k//2][1]),int(cords[k//2][2]),int(cords[k//2][3])))
                                unsafePos.add((int(cords[j//2][0]),int(cords[j//2][1]),int(cords[j//2][2]),int(cords[j//2][3])))
                        else :
                            if k-1 != j :
                                unsafePos.add((int(cords[k//2][0]),int(cords[k//2][1]),int(cords[k//2][2]),int(cords[k//2][3])))
                                unsafePos.add((int(cords[j//2][0]),int(cords[j//2][1]),int(cords[j//2][2]),int(cords[j//2][3])))
            safePos = set(cords)-set(unsafePos)
            for j in unsafePos:
                img1 = cv2.rectangle(img1, (j[0],j[1]), (j[2],j[3]), (0,0,255), 2) #bgr
            for j in safePos:
                img1 = cv2.rectangle(img1, (j[0],j[1]), (j[2],j[3]), (0,255,0), 2)
    frame = cv2.resize(img1, (720,520)) 
    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
vid.release()
cv2.destroyAllWindows()