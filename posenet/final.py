import cv2
from numpy.core.fromnumeric import mean
import pafy
import pandas as pd
# url = "https://www.youtube.com/watch?v=FksYBwUjJZc"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")

video_file = "video/LUNGE_1.mp4"

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import argparse
import posenet
import math
from math import atan2, degrees
# import easydict
# args = easydict.EasyDict({
#     "model":101,
#     "cam_id":best.url,
#     "cam_width":1280,
#     "cam_height":720,
#     "scale_factor":0.712,
#     "file":None
# })

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

angle_dict = {#구하고 싶은 각도들의 인덱스를 저장
    'R hip':(12,6,14),  #오른쪽 옆구리
    'L hip':(11,5,13),  #왼쪽 옆구리
    'R knee':(14,16,12),    #오른쪽 무릎
    'L knee':(13,15,11),     #왼쪽 무릎

    'L pelvis' : (11,13,12),
    'R pelvis' : (12,14,11),
    'L knee' : (13,15,11),
    'R knee' : (14,16,12),
    'L shoulder' : (5,7,11),
    'R shoulder' : (6,8,12),
    'L elbow' : (7,5,9),
    'R elbow' : (8,6,10)
}

def angle_cal(A,B,C):#A에 있는 각도 구하기
    a=math.sqrt((B[0]-C[0])**2+(B[1]-C[1])**2)
    b=math.sqrt((A[0]-C[0])**2+(A[1]-C[1])**2)
    c=math.sqrt((B[0]-A[0])**2+(B[1]-A[1])**2)
    angle=math.acos((b*b+c*c-a*a)/(2*b*c))
    angle=angle*180/math.pi
    return angle

def leg_points(data):
    # data2 = data.iloc[:100,:]
    # max_point = data[data['nose_y']==max(data['nose_y'])].index
    left_ankle_x, left_ankle_y, left_knee_x, left_knee_y ,left_hip_x, left_hip_y = data.loc[:,['leftAnkle_x','leftAnkle_y','leftKnee_x','leftKnee_y', 'leftHip_x','leftHip_y']].iloc[-1]
    right_ankle_x, right_ankle_y, right_knee_x, right_knee_y ,right_hip_x, right_hip_y = data.loc[:,['leftAnkle_x','leftAnkle_y','leftKnee_x','leftKnee_y', 'leftHip_x','leftHip_y']].iloc[-1]
    
    return left_ankle_x, left_ankle_y, left_knee_x, left_knee_y ,left_hip_x, left_hip_y, right_ankle_x, right_ankle_y, right_knee_x, right_knee_y ,right_hip_x, right_hip_y

def shoulder_points(data):
    left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y = data.loc[:,['leftShoulder_x','leftShoulder_y','rightShoulder_x','rightShoulder_y']].iloc[-1]
    return left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y

def body_points(data):
    left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y = data.loc[:,['leftShoulder_x','leftShoulder_y','rightShoulder_x','rightShoulder_y']].iloc[-1]
    return left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y

def get_range(data):
    nose_y, left_ankle_y = data.loc[:,['nose_y','leftAnkle_y']].iloc[-1]
    return nose_y, left_ankle_y

# def get_points_arr(data):
#     p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11 = data.loc[:,['nose_x', 'leftEye_x', 'rightEye_x' 'leftShoulder_x', 'rightShoulder_x', 'leftHip_x', 'rightHip_x', 'leftKnee_x', 'rightKnee_x', 'leftAnkle_x', 'rightAnkle_x']].iloc[-1]
#     point_list = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]
#     return point_list

def angle_between(x1,y1, x2, y2, x3,y3):
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    degree = 360 - (deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2))
    return degree

def torso_area(points):  
    """Return the area of the polygon whose vertices are given by the
    sequence points.
    """
    area = 0
    q = points[-1]
    for p in points:  
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    return area / 2

color = (0,0,250)
test = "Lower"
count_flag = False

def angle_between(x1,y1, x2, y2, x3,y3): #세 x,y로 각도를 구하는 방식
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360
    degree = 360 - (deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2))
    return degree

# ---------- 장연훈 함수 (수정) #########################################################################################################
# ratio: 코와 발목의 y좌표 비율
# grad: 프레임별 ratio의 주기에 따른 그래프의 기울기(+일 경우 내려감, -일 경우 올라옴, 정지하는 부분에서 0에 가까움)
# 무릎과 골반의 y 차이가 가까워질 때
def squat_down(ratio, ratio_arr, grad, squat_knee_angle, left_knee_angle, right_knee_angle, left_hip_gap, right_hip_gap,  standing_ratio):
    global color
    global test
    global count_flag
    global squat_count

    if grad <= -2: # 내려갈 때
        # 스쿼트 판단 하기
        if ratio < ratio_arr[-5]:
            #print('내려갈 때 ratio:', ratio)
            if left_knee_angle > squat_knee_angle * 1.2 and right_knee_angle > squat_knee_angle * 1.2: 
                test="Lower"
                color = (0,0,250)
            elif (squat_knee_angle <= left_knee_angle < squat_knee_angle * 1.2 or 
                    squat_knee_angle <= right_knee_angle < squat_knee_angle * 1.2) and \
                    (left_hip_gap < 10 or right_hip_gap < 10):
                test="Good"
                count_flag = True
                color = (255,0,0)
            
                
    else: # 그냥 서있을때
        if (squat_knee_angle <= left_knee_angle < squat_knee_angle * 1.2 or 
                    squat_knee_angle <= right_knee_angle < squat_knee_angle * 1.2) and \
                    (left_hip_gap < 10 or right_hip_gap < 10):
                test="Good"
                count_flag = True
                color = (255,0,0)
                
        if ratio > ratio_arr[-5] : # 올라갈 때
            print('올라갈 때 ratio 차이:', ratio - ratio_arr[-2])
            test = ""
            color = (0,0,250)

            if ( standing_ratio - 1.0 < ratio ) and ( standing_ratio + 1.0 > ratio ) and count_flag:
                count_flag = False
                print('다시 올라옴')
                squat_count += 1
                # print(squat_count)

    
def to_df(point_x, point_y, x_arr, y_arr):
    x_temp, y_temp = np.array(point_x), np.array(point_y)
    x_arr=np.vstack([x_arr, x_temp])
    y_arr=np.vstack([y_arr, y_temp])
    x_df = pd.DataFrame(x_arr, columns=col_name)
    y_df = pd.DataFrame(y_arr, columns=col_name)
    x_df['frame_num']=range(x_df.shape[0])
    y_df['frame_num']=range(y_df.shape[0])
    mg_df = pd.merge(x_df, y_df, on=['frame_num']).iloc[1:,:]

    return mg_df

parts = ['nose', 'L eye', 'R eye', 'L ear', 'R ear', 
    'L shoulder', 'R shoulder', 'L elbow', 'R elbow', 'L wrist', 'R wrist',
    'L pelvis', 'R pelvis', 'L knee', 'R knee', 'L ankle', 'R ankle']

col_name = ["nose", "leftEye", "rightEye", "leftEar", "rightEar", 
    "leftShoulder","rightShoulder", "leftElbow", "rightElbow", "leftWrist", 
    "rightWrist","leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]



def posenet_search():
    with tf.Session() as sess: #텐서플로우의 세션을 변수에 정의
        model_cfg, model_outputs = posenet.load_model(args.model, sess)  #model_outputs는 텐서 객체들의 리스트
        output_stride = model_cfg['output_stride']
        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(0) # 
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(300/fps)
        start = time.time()
        frame_count = 0
        global squat_count
        squat_count= 0
        peaple_count=1
        bf_keyscores=np.zeros((peaple_count,17),float)
        bf_keycoords=np.zeros((peaple_count,17,2),float)
        min_pose_score=0.3
        min_part_score=0.1
        x_arr = np.zeros((17))
        y_arr = np.zeros((17))
        temp = np.zeros((17))
        ratio_arr = np.array([])
        ratio_arr = []
        grad_check= []
        angle_save={}
        nose_arr = []
        # 프레임별 읽기
        
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride) #영상

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=peaple_count,
                min_pose_score=min_pose_score)

            keypoint_coords *= output_scale

            #---------------------------------------------------------------------------------------------------
            out_img = display_image
            adjacent_keypoints = []
            cv_keypoints = []
            
            # 프레임별 ks, kc 계산
            for ii, score in enumerate(pose_scores):
                for jj,(ks, kc) in enumerate(zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :])):
                    if ks > min_part_score:
                        bf_keyscores[ii][jj]=ks
                        bf_keycoords[ii][jj]=kc
            
            for ii, score in enumerate(pose_scores):
                if score < min_part_score:
                    overlay_image=out_img 
                    bf_keyscores=np.zeros((peaple_count,17),float)
                    bf_keycoords=np.zeros((peaple_count,17,2),float)
                    continue
                results = []
                k_s= bf_keyscores[ii, :]
                k_c= bf_keycoords[ii, :, :]
                
                for left, right in posenet.CONNECTED_PART_INDICES: #선찾기
                    if k_c[left][0] == 0 or k_c[right][1] == 0:
                        continue
                    results.append(np.array([k_c[left][::-1], k_c[right][::-1]]).astype(np.int32),)
                new_keypoints = results
                adjacent_keypoints.extend(new_keypoints)

                # 좌표 받기
                point_x = []
                point_y = []
                
                # 각 부위별로 점을 찍는 곳
                for jj, (ks, kc) in enumerate(zip(k_s, k_c)):#점찾기
                    if kc[0]==0 and kc[0]==1:
                        continue
                    cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
                    point_x.append(kc[1])
                    point_y.append(kc[0])
                    # points = np.c_[points,pos]
                    out_img = cv2.putText(out_img, parts[jj], (int(kc[1]), int(kc[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)
                    if parts[jj] in angle_dict:
                        temp_angle = angle_cal(k_c[angle_dict[parts[jj]][0]], k_c[angle_dict[parts[jj]][1]], k_c[angle_dict[parts[jj]][2]])
                        out_img = cv2.putText(out_img, str(int(temp_angle)), (int(kc[1]), int(kc[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,0,0), 1, cv2.LINE_AA)


                for k,v in angle_dict.items():
                    angle_save[k] = int(angle_cal(k_c[v[0]], k_c[v[1]], k_c[v[2]]))


                mg_df = to_df(point_x, point_y, x_arr, y_arr)
                left_ankle_x, left_ankle_y, left_knee_x, left_knee_y ,left_hip_x, left_hip_y, right_ankle_x, right_ankle_y, right_knee_x, right_knee_y, right_hip_x, right_hip_y = leg_points(mg_df)
                left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y = shoulder_points(mg_df)
                left_knee_angle = angle_between(left_ankle_x, left_ankle_y, left_knee_x, left_knee_y, left_hip_x, left_hip_y) # 왼쪽 무릎 각도
                right_knee_angle = angle_between(right_ankle_x,right_ankle_y, right_knee_x,right_knee_y, right_hip_x, right_hip_y) # 왼쪽 무릎 각도
                left_side_angle = 360 - angle_between(left_knee_x,left_knee_y,left_hip_x, left_hip_y, left_shoulder_x, left_shoulder_y) # 왼쪽 옆구리 각도
                right_side_angle = 360 - angle_between(right_knee_x,right_knee_y,right_hip_x,right_hip_y, right_shoulder_x, right_shoulder_y) # 오른쪽 옆구리 각도
                nose_y, left_ankle_y= get_range(mg_df) # 코 좌표 구하기
                nose_arr.append(nose_y)
                ratio = (left_ankle_y / nose_y)
                ratio_arr.append(ratio)
                
                points = [left_ankle_x, left_ankle_y, left_knee_x, left_knee_y ,left_hip_x, left_hip_y, right_ankle_x, right_ankle_y, right_knee_x, right_knee_y, right_hip_x, right_hip_y, left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y, nose_y]
                temp = np.vstack((temp,points))
            if len(ratio_arr)>20: 
                grad = (mean(nose_arr[-10:-5]) - mean(nose_arr[-5:-1]))/10
                grad_check.append(grad)
                # 자세 판별
                squat_knee_angle = 90
                # left_knee_angle_arr.append(left_knee_angle)
            
            
            left_knee_hip_gap = abs(k_c[11][0] - k_c[13][0])
            right_knee_hip_gap = abs(k_c[12][0] - k_c[14][0])
            standing_ratio = (left_ankle_y / nose_y)
            if len(grad_check)>50:
                # Down ------------------------------------------------------------------------------------------------------------------------
            
                global test
                global color

                squat_down(ratio, ratio_arr,
                            grad, squat_knee_angle, left_knee_angle, right_knee_angle, 
                            left_knee_hip_gap, right_knee_hip_gap,
                            standing_ratio)                
                
                out_img = cv2.putText(out_img, test, (75,100), cv2.FONT_HERSHEY_DUPLEX, 4, color=color, thickness=3)

            out_img = cv2.putText(out_img, str(squat_count), (200,50), cv2.FONT_HERSHEY_PLAIN, 4, color = (255,255,255), thickness=4)
            out_img = cv2.drawKeypoints(
                out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            overlay_image = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
            
            cv2.imshow('posenet', overlay_image)

            cv2.waitKey(delay)
            frame_count += 1
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

        # print(x_arr)
        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    posenet_search()
