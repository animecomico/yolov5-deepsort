import argparse
import os
import sys
from pathlib import Path
from tarfile import PAX_FORMAT

import cv2
import torch
import torch.backends.cudnn as cudnn

from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from models.common import DetectMultiBackend
from utils.datasets import LoadImages, LoadStreams
from utils.general import (LOGGER, non_max_suppression, scale_coords,
                           check_imshow, xyxy2xywh, increment_path)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
import numpy as np
from sklearn.neighbors import KDTree

from scipy.spatial import distance
import pandas as pd
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok, limit_frames, environment, only_detect = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok, opt.limit_frames, \
        opt.environment, opt.only_detect
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    contmax = limit_frames

    contframe = 0
    df_client = pd.DataFrame(columns=['timestamp', 'client', 'frame'])
    df_laptop = pd.DataFrame(columns=['timestamp', 'laptop', 'frame'])
    cont_detect_c = 0
    last_alert = None
    last_alert_laptop = None
    time1 = datetime.now()
    time_client = dict()

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        spf = (datetime.now()-time1).total_seconds()
        time1 = datetime.now()
        print("SPF:", spf)

        contframe = contframe + 1
        if contmax > -1:
            if contmax <= 0:
                break
            contmax = contmax - 1
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string
            if show_vid:
                annotator = Annotator(im0, line_width=2, pil=not ascii)

            # Filter y Region using Centroid Calcultaions
            det_temp = det.clone().detach()
            det2 = det.clone().detach()
            xmin_r = 250
            ymin_r = 250
            xmax_r = 700
            ymax_r = 800
            # Required reescale
            det_temp[:, :4] = scale_coords(img.shape[2:], det_temp[:, :4], im0.shape).round()
            xywhs3 = xyxy2xywh(det_temp[:, 0:4])
            centroid = dict()
            if environment == 'ATM':
                class_filter = 0
            elif environment == 'restaurant':
                class_filter = 1
            else:
                class_filter = 1
            for n_det, detect in enumerate(xywhs3):
                centroid[n_det] = [int((detect[0] + detect[2]) / 2), int((detect[1] + detect[3]) / 2)]
                px = centroid[n_det][0]
                py = centroid[n_det][1]
                valid_on_tegion = False

                if xmin_r <= px <= xmax_r and detect[0] > 5:
                    if ymin_r <= py <= ymax_r:
                        valid_on_tegion = True
                if not valid_on_tegion:
                    det2[n_det] = 10
                else:
                    # David centroid Filter Join
                    for cent in range(n_det):

                        dist = distance.euclidean(centroid[cent], centroid[n_det])
                        if dist < 130 and (det[n_det, -1] + det[cent, -1] == class_filter):
                            if det[n_det, -1] == class_filter:
                                det2[n_det] = 10
                            else:
                                det2[cent] = 10

            det = det2[det2[:, -1] != 10]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                if not only_detect:
                    # pass detections to deepsort
                    t4 = time_sync()
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    clients_pos = []
                    mo_pos = []
                    mv_pos = []
                    id_client = []
                    id_mo = []
                    lappc_pos = []
                    id_lappc = []
                    # Class of models Variables
                    class_client = -1
                    class_mo = -1
                    class_mv = -1
                    class_pc_laptop = -1
                    #
                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            centroid = output[6:]
                            c = int(cls)  # integer class
                            if environment == 'restaurant':
                                class_client = 2
                                class_mo = 0
                                class_mv = 1
                                class_pc_laptop = -1
                            elif environment == 'ATM':
                                class_client = 1
                                class_mo = -1
                                class_mv = -1
                                class_pc_laptop = 0
                            else:
                                print('Not found any environment use default restaurant')
                                class_client = 2
                                class_mo = 0
                                class_mv = 1
                                class_pc_laptop = -1
                            if c == class_client:
                                print(id, centroid)
                                clients_pos.append(centroid)
                                id_client.append(id)
                                if id in time_client:
                                    time_client[id] = time_client[id] + 1*spf
                                else:
                                    time_client[id] = 0

                                print(time_client, contframe)
                            elif c == class_mo:
                                # For now ignore this
                                mo_pos.append(centroid)
                                id_mo.append(id)
                            elif c == class_mv:
                                mv_pos.append(centroid)
                            elif c == class_pc_laptop:
                                lappc_pos.append(centroid)
                                id_lappc.append(id)
                            else:
                                continue
                            if show_vid:
                                label = f'{id} {names[c]} {conf:.2f}'
                                annotator.box_label(bboxes, label, color=colors(c, True))
                        if environment == 'restaurant':
                            enable_client_restaurant = True
                            enable_client_bank = False
                            try:
                                np_mo = np.vstack(mo_pos)
                            except Exception as ero:
                                print('Not Found any class {}'.format(class_mo))
                        elif environment == 'ATM':
                            enable_client_restaurant = False
                            enable_client_bank = True
                        else:
                            print('Use default environment restaurant')
                            enable_client_restaurant = True
                            enable_client_bank = False
                            try:
                                np_mo = np.vstack(mo_pos)
                            except Exception as ero:
                                print('Not Found any class {}'.format(class_mo))
                        try:
                            np_c = np.vstack(clients_pos)
                        except Exception as loi:
                            enable_client_restaurant = False
                            enable_client_bank = False
                        if enable_client_restaurant:
                            print('Client Restaurant Started...')
                            tree = KDTree(np_mo)
                            nearest_dist, nearest_ind = tree.query(np_c, k=2)
                            near_id = nearest_ind[:, 0]
                            near_dist = nearest_dist[:, 0]

                            # print(near_dist)  # drop id; assumes sorted -> see args!
                            # print(near_id)  # drop id
                            # print(id_client)
                            # print(id_mo)
                            num_c = len(id_client)
                            for i in range(0, num_c):
                                client_id = id_client[i]
                                nearid = near_id[i]
                                id_mos = id_mo[nearid]
                                print('Client track id {} on MO track id {}'.format(client_id, id_mos))
                        elif enable_client_bank:
                            if not last_alert is None:
                                if show_vid:
                                    annotator.put_alarm(alarm=last_alert, alarm_on=alarm_on)
                            if not last_alert_laptop is None:
                                if show_vid:
                                    annotator.put_alarm2(alarm=last_alert_laptop, alarm_on=alarm_on_laptop)
                            print('Client Bank Started...')
                            back_prog = 4
                            time_stamp_c = datetime.now()
                            cont_detect_c = cont_detect_c + 1



                            df_client2 = pd.DataFrame(columns=['timestamp', 'client', 'frame'])
                            for idc in id_client:

                                df_client = df_client.append({'client': idc, 'frame': contframe, 'timestamp': time_stamp_c},
                                               ignore_index=True)
                                df_client2 = df_client2.append({'client': idc, 'frame': round(time_client[idc],1), 'timestamp': time_stamp_c},
                                               ignore_index=True)
                                if show_vid:
                                    annotator.print_staytime(df_data=df_client2)
                            for idl in id_lappc:
                                df_laptop = df_laptop.append({'laptop': idl, 'frame': contframe, 'timestamp': time_stamp_c}, ignore_index=True)
                            if cont_detect_c >= back_prog:
                                print('Start Magic')
                            #     Check clients number
                                num_clients = df_client[df_client.duplicated('timestamp', keep=False)]
                                row_c, col_c = num_clients.shape
                                if row_c > 1:
                                    keys = num_clients.index
                                    for i in keys:
                                        time_st = num_clients.loc[i].at['timestamp']
                                        delta_t = time_stamp_c - time_st
                                        days, seconds = delta_t.days, delta_t.seconds
                                        if days == 0 and (2 <= seconds <= 10):
                                            df_client = None
                                            df_client = pd.DataFrame(columns=['timestamp', 'client', 'frame'])
                                            cont_detect_c = 0
                                            print('Alert Clients')
                                            # annotator.put_alarm(alarm='Alert Two Clients')
                                            last_alert = 'Alert Clients'
                                            alarm_on = True
                                            break
                                else:
                                    print('Only One Client')
                                    # annotator.put_alarm(alarm='Only One Client')
                                    last_alert = 'Only One Client'
                                    alarm_on = False
                                if len(lappc_pos) > 0:
                                    keys_p = df_laptop.index
                                    for i in keys_p:
                                        time_st = df_laptop.loc[i].at['timestamp']
                                        delta_t = time_stamp_c - time_st
                                        days, seconds = delta_t.days, delta_t.seconds
                                        if days == 0 and (1 <= seconds <= 10):
                                            df_laptop = None
                                            df_laptop = pd.DataFrame(columns=['timestamp', 'laptop', 'frame'])
                                            cont_detect_c = 0
                                            alarm_on_laptop = True
                                            last_alert_laptop = 'Laptop Alert'
                                            break
                                else:
                                    last_alert_laptop = 'Laptop Alert'
                                    alarm_on_laptop = False
                            else:
                                print('Magic not start yet')
                        else:
                            print('Only Tracking :)')
                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                else:
                    LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s)')
                    if show_vid:
                        bboxes_f = det[:, 0:4].cpu().numpy().astype(int).tolist()
                        confs = confs.cpu().cpu().numpy().tolist()
                        classes = clss.cpu().cpu().numpy().astype(int).tolist()
                        for j, (bboxes, conf, clase) in enumerate(zip(bboxes_f, confs, classes)):
                            print(j, bboxes, conf, clase)
                            c = clase
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')

            if show_vid:
                # Stream results
                im0 = annotator.result()
                if show_vid:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                try:
                    vid_writer.write(im0)
                except Exception as tre:
                    print(repr(tre))

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--limit-frames', type=int, default=-1, help='Number frames limit, -1 no limit')
    parser.add_argument('--environment', type=str, default='restaurant')
    parser.add_argument('--only_detect', action='store_true', help='Only detect objects')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
