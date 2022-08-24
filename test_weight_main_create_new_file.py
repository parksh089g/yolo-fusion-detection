# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640

    추가 내용 퓨전라벨간의 1대1 매칭 확인 알고리즘
    RGB-IR 한번에 검사
    날씨별 mAP weight 곱하는 것 추가
"""
from tqdm import tqdm
import os
import sys
from collections import deque
import numpy as np
from utils.metrics import ap_per_class
from utils.general import LOGGER

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
from sympy import true
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import box_iou, coco80_to_coco91_class, colorstr, check_dataset, check_img_size, \
    check_requirements, check_suffix, check_yaml, increment_path, non_max_suppression, print_args, scale_coords, \
    xyxy2xywh, xywh2xyxy, LOGGER
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def save_one_txt(correct, predn, save_conf, shape, file):
    # Save one txt result
    correct=[cor[0] for cor in correct.tolist()]
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    i=0
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf, correct[i]) if not save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
        i+=1


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=True,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if not save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        data = check_dataset(data)  # check

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    model.half() if half else model.float()

    # Configure
    model.eval()
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=pad, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Run model
        out, train_out = model(img, augment=augment)  # inference and training outputs
        dt[1] += time_sync() - t2

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if not save_txt:
                save_one_txt(correct ,predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])
        #traget이 label, output_to_target()이 pred
        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    # print(stats)
    # print(type(stats))
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    tp, conf, pred_cls, target_cls = stats #tp, conf, pred_cls, target_cls 이때 TP는 iou 0.5~0.95에 따른 값 
    # print('P: ',p)
    # print('r: ',r)
    # print('ap: ',ap)
    # print('f1: ',f1)
    print('TP:',len(tp)) #tp, conf, pred_cls, target_cls
    print('conf: ',len(conf))
    print('pred: ',len(pred_cls))
    print('target: ',len(target_cls))
    with open(save_dir /'target_cls.txt', 'a') as f:
        for i in target_cls:
            f.write(str(int(i))+'\n')
    # for z in range(len(tp)):
    #     print(f'{tp[z]}:::{conf[z]}:::{pred_cls[z]}')

    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t

# output 라벨은 [예측 클래스, x, y, w, h, conf, TP] 순

###############################(RGB 이미지)사용시 수정하세요####################################### 11111
###############################사용시 수정하세요#######################################
def parse_opt_RGB():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / './testing_rgb.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'RGB+ours.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'testing', help='save to project/name')
    parser.add_argument('--name', default='RGB', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt
###############################사용시 수정하세요#######################################   2222222
###############################(IR 이미지)사용시 수정하세요#######################################
def parse_opt_IR():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / './testing_ir.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'IR+ours.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'testing', help='save to project/name')
    parser.add_argument('--name', default='IR', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt
###############################사용시 수정하세요#######################################
def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                LOGGER.info(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, device=opt.device, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_val_study(x=x)  # plot

def obj_matching(weight):
    ###############################사용시 수정하세요#######################################
    ###############################사용시 수정하세요####################################### 333333333
    RGB_weight = {'day' : 0.9,                  #날씨별 mAP 가중치
                      'night': 0.9,
                      'strong_light': 0.9,
                      'rain' : 0.9,
                      'snow' : 0.9 }
    IR_weight = {'day' : 0.9,
                      'night': 0.9,
                      'strong_light': 0.9,
                      'rain' : 0.9,
                      'snow' : 0.9 }
    ###############################사용시 수정하세요#######################################
    ###############################사용시 수정하세요####################################### 44444444
    #기본경로: ~~~ /testing/IR or RGB
    #------------------------------------------------------------------
    ir_label = r"C:\Users\a\Desktop\yolo_fusion\testing\IR\labels" #IR 검출 라벨 경로
    rgb_label = r"C:\Users\a\Desktop\yolo_fusion\testing\RGB\labels" #RGB 검출 라벨 경로
    target_path = r"C:\Users\a\Desktop\yolo_fusion\testing\IR\target_cls.txt" #target_cls.txt 경로
    output_label=r"C:\Users\a\Desktop\yolo_fusion\testing\output" #퓨전 라벨 output 경로
    #------------------------------------------------------------------
    # -----------------------------------------------------------------
    names={0:'person',1:'car'} #class (dictionary 형식) 클래스도 맞게 수정할것!
    save_dir=r"C:\Users\a\Desktop\yolo_fusion\testing\output" #결과 저장 경로
    # -----------------------------------------------------------------
    ###############################사용시 수정하세요#######################################
    ###############################사용시 수정하세요#######################################
    ir_list = os.listdir(ir_label)
    rgb_list = os.listdir(rgb_label)

    if not os.path.exists(output_label):
        os.mkdir(output_label)

    ir = []
    rgb = []
    target_cls = []
    # tp, conf, pred_cls, target_cls = [], [], [], []
    tp_t, conf_t, pred_cls_t, target_cls_t= [], [], [], []
    for i in ir_list:
        if '.txt' in i:
            ir.append(i)
    for r in rgb_list:
        if '.txt' in r:
            rgb.append(r)
    
    if len(ir) != len(rgb):
        print('ir라벨 수 :',len(ir),' rgb라벨 수 :', len(rgb) )
        print('퓨전대상이 불일치합니다. 빈파일을 생성합니다.')
        if len(ir) > len(rgb):
            num = ir
            num2 = rgb_label
            num3 = 'rgb'
        else:
            num = rgb
            num2 = ir_label
            num3 = 'lwir'
        for i in num:
            file=i.split('-')[1]
            rgb_path=num2+'\\'+num3+'-'+file
            # print(rgb_path)
            try:
                rgb_txt=open(rgb_path,'r')
            except:
                print("Create File : ",rgb_path)
                rgb_txt=open(rgb_path,'w')
                # continue
        # sys.exit()
    if True:
        num = ir
        num2 = rgb_label
        num3 = 'rgb'
    for i in num:
        file=i.split('-')[1]
        rgb_path=num2+'\\'+num3+'-'+file
        # print(rgb_path)
        try:
            rgb_txt=open(rgb_path,'r')
        except:
            print('퓨전대상이 불일치합니다. 빈파일을 생성합니다.')
            print("Create File : ",rgb_path)
            rgb_txt=open(rgb_path,'w')
            # sys.exit()
    if True:
        num = rgb
        num2 = ir_label
        num3 = 'lwir'
    for i in num:
        file=i.split('-')[1]
        rgb_path=num2+'\\'+num3+'-'+file
        # print(rgb_path)
        try:
            rgb_txt=open(rgb_path,'r')
        except:
            print('퓨전대상이 불일치합니다. 빈파일을 생성합니다.')
            print("Create File : ",rgb_path)
            rgb_txt=open(rgb_path,'w')
    # sys.exit()
        

    # numbering=0
    for i,r in tqdm(zip(ir,rgb)):
        ir_path=ir_label+'\\'+i
        ir_txt=open(ir_path,'r')
        i_lines=ir_txt.readlines()
        
        rgb_path=rgb_label+'\\'+r
        try:
            rgb_txt=open(rgb_path,'r')
        except:
            continue
        # print(numbering)
        # numbering+=1
        r_lines=deque(rgb_txt.readlines())

        out_path=output_label+'\\'+i
        out_txt=open(out_path,'a')\

        for i_line in i_lines:

            i_pred, i_x, i_y, i_w, i_h, i_conf, i_tp = map(float,i_line.strip().split(' '))

            r_lines2=[]

            while(r_lines): #해당 ir 객체와의 거리 계산
                r_line=r_lines.popleft()
                # r_pred, r_x, r_y, r_w, r_h, r_conf, r_tp = map(float,r_line.strip().split(' '))
                # idx 0    1    2    3    4    5       6
                temp = list(map(float,r_line.strip().split(' ')))

                if not temp[6]: #TP인경우만 계산
                    continue

                temp[3]=(i_x - temp[1])**2+(i_y - temp[2])**2
                r_lines2.append(temp)
            
            # print(r_lines2)
            if len(r_lines2):
                r_lines2=sorted(r_lines2, key=lambda x:x[3])
                r_lines2[0][6]=0

                if i_conf * IR_weight[weight] < r_lines2[0][5] * RGB_weight[weight]:
                    i_conf = r_lines2[0][5]

            else:
                r_lines=False

            tp_t.append([bool(i_tp),False,False,False,False,False,False,False,False,False]) # mAP 0.5 만 계산
            conf_t.append(i_conf)
            pred_cls_t.append(i_pred)

            out_txt.write(f'{i_pred} {i_conf} {i_tp}\n') #입력
        # tp.append(tp_t)
        # conf.append(conf_t)
        # pred_cls.append(pred_cls_t)

    with open(target_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            target_cls.append(int(line))
        f.close()
    # rgb_txt.close()
    # ir_txt.close()
    '''
    첫번째 for문 각각의 파일 하나를 불러옴

    1. 객체매칭 
    r/i_w는 바운딩박스의 너비이지만 AP를 구하는데 필요없으므로 해당 변수에는 객체의 바운딩박스 중심간의 거리를 저장한다.
    (i_x-r_x)^2+(i_y-r_y)^2
    ir 과의 차이를 다구한다. w h 는 의미 없으므로 x y 만 비교
    그중 가장 차이가 작은 거 선택
    conf와 tp 비교 후 대체 혹은 제거
    반복

    ap 계산에 필요한거 TP conf pred_cls target_cls
    '''
    tp=np.array(tp_t)
    conf=np.array(conf_t)
    pred_cls=np.array(pred_cls_t)

    plots=True


    p, r, ap, f1, ap_class = ap_per_class(tp=tp, conf=conf, pred_cls=pred_cls, target_cls=target_cls, plot=plots, save_dir=save_dir, names=names)
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    mp, mr, map50, mAP = p.mean(), r.mean(), ap50.mean(), ap.mean()

    print( '1. mp : ', mp)
    print( '2. mr : ',mr)
    print( '3. map50 : ',map50)
    print( '4. map(미사용): ',mAP)

    # tp, conf, pred_cls, target_cls = stats #tp, conf, pred_cls, target_cls 이때 TP는 iou 0.5~0.95에 따른 값 

    # pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    # LOGGER.info(pf % ('all', mp, mr, map50, map))
    # # LOGGER.info(pf % ('all', mp, mr, map50))

    # for i, c in enumerate(ap_class):
    #     LOGGER.info(pf % (names[c], p[i], r[i], ap50[i], ap[i]))

if __name__ == "__main__":
    opt_RGB = parse_opt_RGB()
    main(opt_RGB)
    opt_IR = parse_opt_IR()
    main(opt_IR)
    ################################################ 555555555
    weight= 'day' #날씨 입력 'day' / 'night' / 'strong_light' / 'rain' / 'snow'
    ################################################
    obj_matching(weight)


    ### 주석에 달린 1~5번 까지 수정 및 확인할것 