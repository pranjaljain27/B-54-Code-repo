# python detect_track_UI_v8_centerstage.py --source ../8_2.mp4 --cfg cfg/yolor_p6.cfg --weights yolor_p6.pt --conf 0.25 --img-size 1280 --view-img
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import re
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

# 1. define window size and variables for zoom, center, zoom_in, dynamic width
speed_four_line_queue = {}
window_size_height = 247
window_size_width = 440
dynamic_width = 640
zoom_counter_x = 0
zoom_counter_y = 0
zoom_in_required_flag = 0
center_stage_center_x = None
center_stage_center_y = None

top_right_coord_y = 0
top_left_coord_x = 0

bottom_right_coord_y = window_size_height
bottom_right_coord_x = window_size_width

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person  #BGR
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        
        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Maintain fix size window and perform crop using function "find_centerstage_window"# steps:
def find_centerstage_window(img_copy, center_x, center_y, width, height, zoom_counter_x, zoom_counter_y, dynamic_width):
    top_left_coord_x = center_x - (window_size_width//2) -zoom_counter_x
    top_right_coord_y = center_y - (window_size_height//2) -zoom_counter_y

    bottom_right_coord_x =  center_x + (window_size_width//2) + zoom_counter_x #if center_x + window_size_width < wd else wd
    bottom_right_coord_y = center_y + (window_size_height//2) + zoom_counter_y#if center_y + window_size_height < ht else ht

    #If center point is at corner, we try to find fix size window
    if top_left_coord_x < 0:
        bottom_right_coord_x -= top_left_coord_x #update our bottom right coordinate as well
        top_left_coord_x = 0

    if top_right_coord_y < 0:
        bottom_right_coord_y -= top_right_coord_y
        top_right_coord_y = 0

    if bottom_right_coord_x > width:
        diff_x = bottom_right_coord_x - width
        top_left_coord_x -= diff_x
        bottom_right_coord_x = width

    if bottom_right_coord_y > height:
        diff_y = bottom_right_coord_y - height
        top_right_coord_y -= diff_y
        bottom_right_coord_y = height

    bottom_right_coord = (bottom_right_coord_x, bottom_right_coord_y) 

    cropped_img = img_copy[top_right_coord_y:bottom_right_coord_y, top_left_coord_x:bottom_right_coord_x]

    cv2.rectangle(img_copy, (top_left_coord_x, top_right_coord_y), bottom_right_coord, (0, 255, 0), 2) 

    dynamic_width = bottom_right_coord_x - top_left_coord_x
    cropped_img = cv2.resize(cropped_img, (window_size_width, window_size_height)) 
    return cropped_img, (top_left_coord_x, top_right_coord_y), bottom_right_coord, dynamic_width

# main drawing function
def draw_boxes_center_stage(img, bbox, object_id, identities=None, offset=(0, 0)):

    # access variable globally (accessed from anywhere in your program)
    global dynamic_width
    global zoom_counter_x
    global zoom_counter_y
    global zoom_in_required_flag
    global center_stage_center_x
    global center_stage_center_y

    height, width, _ = img.shape

    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    person_locations = []

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]  #Assigning Coordinates
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # Find center of object
        obj_center = (int((x2+x1)/ 2), int((y1+y1)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= opt.trailslen)
          speed_four_line_queue[id] = [] 

        #Assign label with cooresponding class color
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '%s' % (obj_name)
        UI_box(box, img, label=label, color=color, line_thickness=2)

        if obj_name == "person": 
            person_locations.append([x1, y1, x2, y2])
                    
        # add center to buffer
        data_deque[id].appendleft(center)

        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue

            # generate dynamic thickness of trails
            thickness = int(np.sqrt(opt.trailslen / float(i + i)) * 1.5)
            
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    img_copy = img.copy()
    
    # 2. get a center point for 1 or 2 people using YoloR

    print(person_locations)
    
    if len(person_locations) > 0:
        if len(person_locations) >1:
            persons_np = np.array(person_locations)
            m1 = np.min(persons_np,axis=0)
            n1 = m1[1]  #M is for x axis
            m1 = m1[0]  #N is for y axis
            m2 = np.max(persons_np,axis=0)
            n2 = m2[3]
            m2 = m2[2]

            center_x = (m1 + m2)//2 
            center_y = (n1 + ((n2 - n1)//4))

            xlen = m2 - m1 
            print(xlen)
            print(dynamic_width)
        else: # just get location of the only person detected.
            m1,n1,m2,n2 = person_locations[0] 
            center_x = (m1 + m2)//2 
            center_y = (n1 + ((n2 - n1)//4))

            xlen = m2 - m1 #Width of object
            print(xlen)

    try:
        # 3. Compare center with the last center if it's changed then move window towards a new point, set speed towards new point by changing 16//4 for the x-axis and 9//4 for the y-axis
        # update center
        if center_stage_center_x == None:
            center_stage_center_x = center_x
        else:
            # 4. 16:9 is an aspect ratio of the video so please select in multiplication of 16 and 9 

            if center_stage_center_x <= center_x:
                dis_x = center_x - center_stage_center_x
                # from here we can change speed of user tracking if you use 16//2 or 16... it will follow object very fast  
                if dis_x >= 16//4:
                    center_stage_center_x += 16//4  #update speed in x
                else:
                    center_stage_center_x = center_x
            else:
                dis_x = center_stage_center_x - center_x
                if dis_x >= 16//4:
                    center_stage_center_x -= 16//4
                else:
                    center_stage_center_x = center_x

        if center_stage_center_y == None:
            center_stage_center_y = center_y   
        else:
            # 4. 16:9 is an aspect ratio of the video so please select in multiplication of 16 and 9 

            if center_stage_center_y <= center_y:
                dis_x = center_y - center_stage_center_y #update speed in y
                if dis_x >= 9//4:
                    center_stage_center_y += 9//4
                else:
                    center_stage_center_y = center_y
            else:
                dis_x = center_stage_center_y - center_y

                # from here we can change speed of user tracking if you use 9//2 or 9... it will follow object very fast  
                if dis_x >= 9//4:
                    center_stage_center_y -= 9//4
                else:
                    center_stage_center_y = center_y


        # 5. perform zoom out if object size is more than crop window size using dynamic width
        # else 5. perform zoom in after every 20 frame if required "if zoom_in_required_flag % 20 == 0"

        if xlen > dynamic_width:
            zoom_counter_x += 16
            zoom_counter_y += 9
        elif xlen < dynamic_width and dynamic_width > window_size_width - 16:            
            zoom_in_required_flag +=1
            if zoom_in_required_flag % 16 == 0: #perform zoom in after every 16 frames
                zoom_counter_x -= 16//4
                zoom_counter_y -= 9//4
        
        # 6. Maintain fix size window and perform crop using function "find_centerstage_window"# steps:
        new_cropped, _,_, dynamic_width = find_centerstage_window(img, center_stage_center_x, center_stage_center_y, width, height, zoom_counter_x, zoom_counter_y, dynamic_width)

    except Exception as e:
        print(e)
        new_cropped = img_copy[top_right_coord_y:bottom_right_coord_y, top_left_coord_x:bottom_right_coord_x]

    return img, new_cropped


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)


    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = Darknet(cfg, imgsz)#.cuda()
    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)

    # Get names and colors
    names = load_classes(names)
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    prevTime = 0
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywh_bboxs = []
                confs = []
                oids = []
                # Write results
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    oids.append(int(cls))

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                
                outputs = deepsort.update(xywhs, confss, oids, im0)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]

                    im0, new_cropped = draw_boxes_center_stage(im0, bbox_xyxy, object_id,identities) ##
                    print(im0.shape)
                else:
                    new_cropped = im0[top_right_coord_y:bottom_right_coord_y, top_left_coord_x:bottom_right_coord_x]##
                    
            else:
                new_cropped = img[top_right_coord_y:bottom_right_coord_y, top_left_coord_x:bottom_right_coord_x]##
                
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            
            cv2.line(im0, (20,25), (127,25), [85,45,255], 30)
            cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)


            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                cv2.imshow("CROP_WINDOW", new_cropped)
                
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w,h))#(window_size_width, window_size_height))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    parser.add_argument('--trailslen', type=int, default=64, help='trails size (new parameter)')
        
    opt = parser.parse_args()
    print(opt)
    global names
    names = load_classes(opt.names)
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
