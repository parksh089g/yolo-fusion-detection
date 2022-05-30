from tqdm import tqdm
import os
import sys
from collections import deque
import numpy as np
from utils.metrics import ap_per_class
from utils.general import LOGGER
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#2022 - 05 - 30 - 최종수정본
#------------------------------------------------------------------
ir_label = r"C:\Users\a\Desktop\yolo_fusion\testing\RGB_Rain\labels" #IR 검출 라벨 경로 <== 기준라벨 (타겟과 같아야함)
rgb_label = r"C:\Users\a\Desktop\yolo_fusion\testing\IR_Rain\labels" #RGB 검출 라벨 경로
target_path = r"C:\Users\a\Desktop\yolo_fusion\testing\RGB_Rain\target_cls.txt" #target_cls.txt 경로
output_label=r"C:\Users\a\Desktop\yolo_fusion\testing\output_" #퓨전 라벨 output 경로
#------------------------------------------------------------------
# -----------------------------------------------------------------
names={0:'person',1:'car'} #class (dictionary 형식)
save_dir=r"C:\Users\a\Desktop\yolo_fusion\testing\output_" #결과 저장 경로
# -----------------------------------------------------------------

ir_list = os.listdir(ir_label)
rgb_list = os.listdir(rgb_label)

if not os.path.exists(output_label):
    os.mkdir(output_label)
# print(len(ir_list),len(rgb_list))
if len(ir_list) != len(rgb_list):
    print('ir과 rgb 라벨수가 다릅니다! 실행을 중지합니다.')
    sys.exit()

ir = []
rgb = []
target_cls = []
# tp, conf, pred_cls, target_cls = [], [], [], []
tp_t, conf_t, pred_cls_t, target_cls_t= [], [], [], []
for i, r in zip(ir_list,rgb_list):
    if '.txt' in i:
        ir.append(i)
    if '.txt' in r:
        rgb.append(r)


for i,r in tqdm(zip(ir,rgb)):
    ir_path=ir_label+'\\'+i
    ir_txt=open(ir_path,'r')
    i_lines=ir_txt.readlines()
    fname=i.split('-')[1]
    # rgb_path=rgb_label+'\\'+'lwir-'+fname #-------------------------------- lwir 또는 rgb로 바꾸세요
    rgb_path=rgb_label+'\\'+r
    if not os.path.exists(rgb_path):
        print("x")
        continue

    rgb_txt=open(rgb_path,'r')
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

            if i_conf < r_lines2[0][5]:
                i_conf = r_lines2[0][5]
                i_tp=True

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
rgb_txt.close()
ir_txt.close()
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
mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

print( '1. mp : ', mp)
print( '2. mr : ',mr)
print( '3. map50 : ',map50)
print( '4. map(미사용): ',map)

# tp, conf, pred_cls, target_cls = stats #tp, conf, pred_cls, target_cls 이때 TP는 iou 0.5~0.95에 따른 값 

# pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
# LOGGER.info(pf % ('all', mp, mr, map50, map))
# # LOGGER.info(pf % ('all', mp, mr, map50))

# for i, c in enumerate(ap_class):
#     LOGGER.info(pf % (names[c], p[i], r[i], ap50[i], ap[i]))