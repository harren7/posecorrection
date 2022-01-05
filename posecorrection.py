import json
import trt_pose.coco
import trt_pose.models
import torch
import torch2trt
from torch2trt import TRTModule
import time, sys
import cv2
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import argparse
import os.path
import numpy as np


'''
hnum: 0 based human index
kpoint : keypoints (float type range : 0.0 ~ 1.0 ==> later multiply by image width, height
'''

#Pose Functions
def check_slump(neck_y,nose_y):
    if neck_y != -1 and nose_y != -1 and nose_y > neck_y - 0.2 :
        return False # Pose Detected and the pose is unbalanced
    elif neck_y != -1 and nose_y != -1 and nose_y <= neck_y - 0.2  :
        return 0  # Pose Detected and the pose is balanced
    else :
        return True # Nothing Detected

def check_tilted_left(leftshoulder_x, leftear_x):
    if leftshoulder_x != -1 and leftear_x != -1 and leftshoulder_x > leftear_x :
        return False
    elif leftshoulder_x != -1 and leftear_x != -1 and leftshoulder_x <= leftear_x :
        return 0
    else :
        return True

def check_tilted_right(rightshoulder_x, rightear_x):
    if rightshoulder_x != -1 and rightear_x != -1 and rightshoulder_x < rightear_x :
        return False
    elif rightshoulder_x != -1 and rightear_x != -1 and rightshoulder_x >= rightear_x:
        return 0 
    else :
        return True

def check_tilted_pelvis(leftpelvis_y, rightpelvis_y):
    if leftpelvis_y != -1 and rightpelvis_y != -1 and not -0.2 <=(leftpelvis_y - rightpelvis_y)<=0.2 :
        return False
    elif  leftpelvis_y != -1 and rightpelvis_y != -1 and -0.2 <=(leftpelvis_y - rightpelvis_y)<=0.2 :
        return 0
    else :
        return True

def check_knee(leftknee_y, rightknee_y) :
    if leftknee_y != -1 and rightknee_y != -1 and not -0.4<= (leftknee_y - rightknee_y)<=0.4 :
        return False
    elif leftknee_y != -1 and rightknee_y != -1 and -0.4<= (leftknee_y - rightknee_y)<=0.4 :
        return 0
    else :
        return True

def check_ankle(leftankle_y, rightankle_y) :
    if leftankle_y != -1 and rightankle_y != -1 and not -0.2 <=(leftankle_y - rightankle_y) <= 0.2 :
        return False
    elif leftpelvis_y != -1 and rightpelvis_y != -1 and -0.2 <=(leftankle_y - rightankle_y) <= 0.2 :
        return 0
    else :
        return True

def check_headdrop(lefteye_y, leftear_y) :
    if lefteye_y != -1 and leftear_y != -1 and lefteye_y > leftear_y :
        return False
    elif leftpelvis_y != -1 and rightpelvis_y != -1 and lefteye_y <= leftear_y :
        return 0
    else :
        return True

str_slump = "Sitting Right :  "
str_tiltedl = "Left Head and Shoulder :   " 
str_tiltedr = "Right Head and Shoulder :   "
str_pelvis = "Pelvis Balance :   " 
str_knee = "Leg Posture :  " 
str_ankle = "Standing straight on both foot :  "
str_head = "Head is Up :   "

def default_message() :
    str_list = str_slump + str_tiltedl  + str_tiltedr + str_pelvis + str_knee + str_ankle + str_head

def get_keypoint(humans, hnum, peaks):
    #check invalid human index
    kpoint = []
    human = humans[0][hnum]
    C = human.shape[0]
    #create list to save 17 feature y,x values
    feature_list = np.zeros([18,2])
    for j in range(C):
        k = int(human[j])
        if k >= 0:
            peak = peaks[0][j][k]   # peak[1]:width, peak[0]:height
            peak = (j, float(peak[0]), float(peak[1]))
            kpoint.append(peak)
            #save feature y,x in 3d list
            feature_list[j][0] = peak[1]
            feature_list[j][1] = peak[2]
            #print('feature list is : ', feature_list)
            #print feature y,x 
            #print('index:%d : success [%5.3f, %5.3f]'%(j, peak[1], peak[2]) )

        else:
            peak = (j, None, None)
            kpoint.append(peak)
            #print('index:%d : None %d'%(j, k) )

        #Print X if the posture is wrong, and print O if it's okay
        if not check_slump(feature_list[17][0],feature_list[0][0]):
            slump = str_slump + "X \n"
            print(slump)
        elif check_slump(feature_list[17][0],feature_list[0][0]) == 0 :
            a = str_slump + "O \n"
            print(slump)
        elif check_slump(feature_list[17][0],feature_list[0][0]) :
            a = str_slump
            print(slump)

        if not check_tilted_left(feature_list[6][1], feature_list[4][1]):
            tiltedl = str_tiltedl + "X \n"
            print(tiltedl)
        elif check_tilted_left(feature_list[6][1], feature_list[4][1]) == 0 :
            tiltedl= str_tiltedl + "O \n"
            print(tiltedl)
        elif check_tilted_left(feature_list[6][1], feature_list[4][1]) :
            tiltedl = str_tiltedl
            print(tiltedl)
            
        if not check_tilted_right(feature_list[5][1], feature_list[3][1]):
            tiltedr = str_tiltedr + "X \n"
            print(tiltedr)
        elif check_tilted_right(feature_list[5][1], feature_list[3][1]) == 0 :
            tiltedr = str_tiltedr + "O \n"
            print(tiltedr)
        elif check_tilted_right(feature_list[5][1], feature_list[3][1]) :
            tiltedr = str_tiltedr
            print(tiltedr)

        if not check_tilted_pelvis(feature_list[12][0], feature_list[11][0]):
            pelvis = str_pelvis + "X \n"
            print(pelvis)
        elif check_slump(feature_list[17][0],feature_list[0][0]) == 0 :
            pelvis = str_pelvis + "O \n"
            print(pelvis)
        elif check_slump(feature_list[17][0],feature_list[0][0]) :
            pelvis = str_pelvis
            print(pelvis)

        if not check_knee(feature_list[14][1],feature_list[13][1]):
            knee = str_knee + "X \n"
            print(knee)
        elif check_slump(feature_list[17][0],feature_list[0][0]) == 0 :
            knee = str_knee + "O \n"
            print(knee)
        elif check_slump(feature_list[17][0],feature_list[0][0]) :
            knee = str_knee 
            print(knee)
            
        if not check_ankle(feature_list[16][1], feature_list[15][1]):
            ankle = str_ankle + "X \n"
            print(ankle)
        elif check_slump(feature_list[17][0],feature_list[0][0]) == 0 :
            ankle = str_ankle + "O \n"
            print(ankle)
        elif check_slump(feature_list[17][0],feature_list[0][0]) :
            ankle = str_ankle
            print(ankle)
            
        if not check_headdrop(feature_list[2][0], feature_list[4][0]):
            head = str_head + "X \n"
            print(head)
        elif check_slump(feature_list[17][0],feature_list[0][0]) == 0 :
            head = str_head + "O \n"
            print(head)
        elif check_slump(feature_list[17][0],feature_list[0][0]) :
            head = str_head
            print(head)

        all_messages = slump + tiltedl + tiltedr + pelvis +  knee + ankle + head
            
    return kpoint, feature_list, all_messages

parser = argparse.ArgumentParser(description='TensorRT pose estimation run')
parser.add_argument('--model', type=str, default='resnet', help = 'resnet or densenet' )
args = parser.parse_args()

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])


if 'resnet' in args.model:
    print('------ model = resnet--------')
    MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
    OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 224
    HEIGHT = 224

else:
    print('------ model = densenet--------')
    MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'
    OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'


    model = trt_pose.models.densenet121_baseline_att(num_parts, 2 * num_links).cuda().eval()
    WIDTH = 256
    HEIGHT = 256

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()
if os.path.exists(OPTIMIZED_MODEL) == False:
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1<<25)
    torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

t0 = time.time()
torch.cuda.current_stream().synchronize()
for i in range(50):
    y = model_trt(data)
torch.cuda.current_stream().synchronize()
t1 = time.time()

print(50.0 / (t1 - t0))

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)

    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def execute(img, src, t):
    color = (0, 255, 0)
    data = preprocess(img)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    fps = 1.0 / (time.time() - t)
    for i in range(counts[0]):
        keypoints,feature__list, message__list = get_keypoint(objects, i, peaks)
        for j in range(len(keypoints)):
            if keypoints[j][1]:
                x = round(keypoints[j][2] * WIDTH * X_compress)
                y = round(keypoints[j][1] * HEIGHT * Y_compress)
                cv2.circle(src, (x, y), 3, color, 2)
                cv2.putText(src , "%d" % int(keypoints[j][0]), (x + 5, y),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
                cv2.circle(src, (x, y), 3, color, 2)
                cv2.putText
                #cv2.putText(src, "%.4f %.4f" % (feature__list[j][0], feature__list[j][1]), (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1)

    print("FPS:%f "%(fps))
    #draw_objects(img, counts, objects, peaks)

    cv2.putText(src , "FPS: %f" % (fps), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(src , "%s" % (message__list), (20, 150),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
    out_video.write(src)
    cv2.imshow('key', src)

cap_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)NV12, framerate=(fraction)24/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
cap = cv2.VideoCapture(cap_str)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret_val, img = cap.read()
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_video = cv2.VideoWriter('/tmp/output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, 480))
count = 0

X_compress = 640.0 / WIDTH * 1.0
Y_compress = 480.0 / HEIGHT * 1.0

if cap is None:
    print("Camera Open Error")
    sys.exit(0)

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

while cap.isOpened() and count < 500:
    t = time.time()
    ret_val, dst = cap.read()
    if ret_val == False:
        print("Camera read Error")
        break

    img = cv2.resize(dst, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    execute(img, dst, t)

    count += 1


cv2.destroyAllWindows()
out_video.release()
cap.release()
