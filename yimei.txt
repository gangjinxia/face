import os
# 屏蔽MediaPipe底层冗余日志
os.environ['GLOG_minloglevel'] = '2'

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 人脸检测器初始化
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 468关键点索引
PTS_IDX = {
    "left_eye_left": 33, "left_eye_right": 133,
    "right_eye_left": 362, "right_eye_right": 263,
    "top_forehead": 10, "bottom_chin": 152,
    "left_cheek": 234, "right_cheek": 454,
    "nose_top": 1, "nose_bottom": 4,
    "lip_top": 0, "lip_bottom": 17,
    "left_jaw": 234, "right_jaw": 454,
    "brow_top": 9
}

class BeautyFaceAnalyzer:
    def __init__(self, img_bgr: np.ndarray):
        self.img = img_bgr
        self.h, self.w = img_bgr.shape[:2]
        self.landmarks_2d: np.ndarray = None

    def detect_face_mesh(self):
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = detector.detect(mp_image)
        if not res.face_landmarks:
            return False
        lm_list = res.face_landmarks[0]
        pts = []
        for lm in lm_list:
            x = lm.x * self.w
            y = lm.y * self.h
            pts.append([x, y])
        self.landmarks_2d = np.array(pts, dtype=np.float32)
        return True

    def get_point(self, idx):
        return self.landmarks_2d[idx]

    def calc_three_court_five_eye(self):
        fore_top = self.get_point(PTS_IDX["top_forehead"])
        nose_mid = self.get_point(PTS_IDX["nose_bottom"])
        chin_bottom = self.get_point(PTS_IDX["bottom_chin"])
        brow_top = self.get_point(PTS_IDX["brow_top"])

        le_l = self.get_point(PTS_IDX["left_eye_left"])
        le_r = self.get_point(PTS_IDX["left_eye_right"])
        re_l = self.get_point(PTS_IDX["right_eye_left"])
        re_r = self.get_point(PTS_IDX["right_eye_right"])

        mid_court = brow_top[1] - fore_top[1]
        upper_court = nose_mid[1] - brow_top[1]
        lower_court = chin_bottom[1] - nose_mid[1]
        total_h = chin_bottom[1] - fore_top[1]

        ratio_up = round(float(mid_court / total_h), 3)
        ratio_mid = round(float(upper_court / total_h), 3)
        ratio_low = round(float(lower_court / total_h), 3)

        eye_w = np.linalg.norm(le_r - le_l)
        face_w = np.linalg.norm(self.get_point(PTS_IDX["right_cheek"]) - self.get_point(PTS_IDX["left_cheek"]))
        eye_ratio = round(float((eye_w * 5) / face_w), 3)

        return {
            "三庭比例(上/中/下)": [ratio_up, ratio_mid, ratio_low],
            "五眼匹配度(理想值=1)": eye_ratio
        }

    def calc_symmetry_score(self):
        mid_x = self.get_point(PTS_IDX["nose_top"])[0]
        total = 0
        pairs = [("left_eye_left", "right_eye_right"), ("left_cheek", "right_cheek"), ("left_jaw", "right_jaw")]
        for lk, rk in pairs:
            lp = self.get_point(PTS_IDX[lk])
            rp = self.get_point(PTS_IDX[rk])
            d1 = abs(mid_x - lp[0])
            d2 = abs(rp[0] - mid_x)
            diff = abs(d1 - d2)
            total += 1 / (1 + diff / self.w)
        score = round(float(total / len(pairs)), 3)
        return {"面部对称得分(满分1)": score}

    def skin_region_analysis(self):
        mask = np.zeros((self.h, self.w), dtype=np.uint8)
        jaw_idx = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
        contour = self.landmarks_2d[jaw_idx].astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)
        skin_bgr = cv2.bitwise_and(self.img, self.img, mask=mask)
        skin_hsv = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2HSV)
        total_skin = np.count_nonzero(mask)
        if total_skin == 0:
            return {"泛红面积占比":0.0, "暗沉面积占比":0.0}

        red_mask = cv2.inRange(skin_hsv, (0,80,50), (20,255,255))
        red_pix = np.count_nonzero(red_mask & mask)
        red_ratio = round(float(red_pix / total_skin), 3)

        dark_mask = cv2.inRange(skin_hsv, (0,0,0), (180,255,90))
        dark_pix = np.count_nonzero(dark_mask & mask)
        dark_ratio = round(float(dark_pix / total_skin), 3)

        return {"泛红面积占比":red_ratio, "暗沉面积占比":dark_ratio}

    def draw_all_landmark(self):
        draw_img = self.img.copy()
        pts_int = self.landmarks_2d.astype(np.int32)
        for x,y in pts_int:
            cv2.circle(draw_img, (x,y), 1, (0,255,0), -1)
        jaw_idx = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
        contour = self.landmarks_2d[jaw_idx].astype(np.int32)
        cv2.polylines(draw_img, [contour], True, (0,0,255), 1)
        return draw_img

if __name__ == "__main__":
    image = cv2.imread("wanqian.webp")
    if image is None:
        print("图片读取失败，请把 wanqian.webp 放到代码同目录")
    else:
        analyzer = BeautyFaceAnalyzer(image)
        if not analyzer.detect_face_mesh():
            print("未检测到人脸，请使用清晰正面人像")
        else:
            print("===== AI医美面诊分析报告 =====")
            res1 = analyzer.calc_three_court_five_eye()
            res2 = analyzer.calc_symmetry_score()
            res3 = analyzer.skin_region_analysis()
            print("三庭五眼：", res1)
            print("面部对称：", res2)
            print("皮肤检测：", res3)
            show_img = analyzer.draw_all_landmark()
            cv2.imshow("医美人脸网格分析", show_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()