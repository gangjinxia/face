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
        gray = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2GRAY)
        total_skin = np.count_nonzero(mask)
        if total_skin == 0:
            return {
                "泛红面积占比":0.0, "暗沉面积占比":0.0,
                "色斑色素占比":0.0, "毛孔粗糙占比":0.0,
                "出油高光占比":0.0, "干纹缺水占比":0.0
            }

        # 1.泛红
        red_mask = cv2.inRange(skin_hsv, (0,80,50), (20,255,255))
        red_pix = np.count_nonzero(red_mask & mask)
        red_ratio = round(float(red_pix / total_skin), 3)

        # 2.暗沉
        dark_mask = cv2.inRange(skin_hsv, (0,0,0), (180,255,90))
        dark_pix = np.count_nonzero(dark_mask & mask)
        dark_ratio = round(float(dark_pix / total_skin), 3)

        # 3.色斑黄褐色
        spot_mask = cv2.inRange(skin_hsv, (10,40,40), (35,160,160))
        spot_pix = np.count_nonzero(spot_mask & mask)
        spot_ratio = round(float(spot_pix / total_skin), 3)

        # 4.出油高光
        oil_mask = cv2.inRange(skin_hsv, (0,0,210), (180,40,255))
        oil_pix = np.count_nonzero(oil_mask & mask)
        oil_ratio = round(float(oil_pix / total_skin), 3)

        # 5.干纹缺水细纹
        dry_line_mask = cv2.inRange(skin_hsv, (0,0,60), (180,70,130))
        dry_pix = np.count_nonzero(dry_line_mask & mask)
        dry_ratio = round(float(dry_pix / total_skin), 3)

        # 6.毛孔粗糙纹理
        blur_gray = cv2.GaussianBlur(gray, (3,3), 0)
        pore_mask = cv2.subtract(blur_gray, gray)
        _, pore_bin = cv2.threshold(pore_mask, 12, 255, cv2.THRESH_BINARY)
        pore_pix = np.count_nonzero(pore_bin & mask)
        pore_ratio = round(float(pore_pix / total_skin), 3)

        return {
            "泛红面积占比": red_ratio,
            "暗沉面积占比": dark_ratio,
            "色斑色素占比": spot_ratio,
            "毛孔粗糙占比": pore_ratio,
            "出油高光占比": oil_ratio,
            "干纹缺水占比": dry_ratio
        }

    def calc_eye_nose_lip(self):
        le_l = self.get_point(33)
        le_r = self.get_point(133)
        re_l = self.get_point(362)
        re_r = self.get_point(263)
        left_eye_top = self.get_point(145)
        left_eye_bottom = self.get_point(159)
        right_eye_top = self.get_point(374)
        right_eye_bottom = self.get_point(386)

        left_eye_h = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_h = abs(right_eye_top[1] - right_eye_bottom[1])
        inter_eye_dist = abs(re_l[0] - le_r[0])
        eye_avg_w = (np.linalg.norm(le_r - le_l) + np.linalg.norm(re_r - re_l)) / 2

        forehead_mid = self.get_point(10)
        nose_tip = self.get_point(4)
        nose_left_wing = self.get_point(234)
        nose_right_wing = self.get_point(454)
        nose_width = np.linalg.norm(nose_right_wing - nose_left_wing)
        nose_total_vertical = abs(nose_tip[1] - forehead_mid[1])

        lip_left = self.get_point(61)
        lip_right = self.get_point(291)
        lip_top_mid = self.get_point(0)
        lip_bottom_mid = self.get_point(17)
        lip_cupids_bow = self.get_point(13)
        lip_bottom_center = self.get_point(14)
        philtrum_top = self.get_point(1)

        lip_w = np.linalg.norm(lip_right - lip_left)
        upper_lip_h = abs(lip_top_mid[1] - lip_cupids_bow[1])
        lower_lip_h = abs(lip_bottom_mid[1] - lip_bottom_center[1])
        philtrum = abs(lip_cupids_bow[1] - philtrum_top[1])

        return {
            "平均眼宽(占脸宽)": round(float(eye_avg_w / self.w), 3),
            "眼间距(占脸宽)": round(float(inter_eye_dist / self.w), 3),
            "单眼平均高度(占脸高)": round(float((left_eye_h + right_eye_h) / 2 / self.h), 3),
            "鼻翼宽度(占脸宽)": round(float(nose_width / self.w), 3),
            "鼻长(眉心至鼻尖/占脸高)": round(float(nose_total_vertical / self.h), 3),
            "嘴唇宽度(占脸宽)": round(float(lip_w / self.w), 3),
            "上唇厚度(占脸高)": round(float(upper_lip_h / self.h), 3),
            "下唇厚度(占脸高)": round(float(lower_lip_h / self.h), 3),
            "人中长度(占脸高)": round(float(philtrum / self.h), 3)
        }

    def calc_face_shape(self):
        cheek_left = self.get_point(234)
        cheek_right = self.get_point(454)
        jaw_left = self.get_point(58)
        jaw_right = self.get_point(291)
        top_fore = self.get_point(10)
        bottom_chin = self.get_point(152)

        cheek_width = np.linalg.norm(cheek_right - cheek_left)
        jaw_width = np.linalg.norm(jaw_right - jaw_left)
        face_length = abs(bottom_chin[1] - top_fore[1])

        length_cheek_ratio = face_length / cheek_width
        jaw_cheek_ratio = jaw_width / cheek_width

        face_type = "鹅蛋脸"
        if length_cheek_ratio > 1.4:
            face_type = "长脸"
        elif jaw_cheek_ratio > 0.92 and length_cheek_ratio < 1.25:
            face_type = "方脸"
        elif jaw_cheek_ratio < 0.78:
            face_type = "菱形脸"
        elif length_cheek_ratio < 1.15:
            face_type = "圆脸"

        return {
            "脸型分类": face_type,
            "脸长/颧骨宽比值": round(float(length_cheek_ratio), 3),
            "下颌宽/颧骨宽比值": round(float(jaw_cheek_ratio), 3)
        }

    def draw_all_landmark(self):
        draw_img = self.img.copy()
        pts_int = self.landmarks_2d.astype(np.int32)

        for x, y in pts_int:
            cv2.circle(draw_img, (x, y), 1, (0, 255, 0), -1)

        jaw_idx = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
        contour = self.landmarks_2d[jaw_idx].astype(np.int32)
        cv2.polylines(draw_img, [contour], True, (0, 0, 255), 1)

        h, w = self.h, self.w
        fore_top_pt = self.get_point(PTS_IDX["top_forehead"])
        chin_pt = self.get_point(PTS_IDX["bottom_chin"])

        mid_x = int(self.get_point(PTS_IDX["nose_top"])[0])
        y_min = int(fore_top_pt[1])
        y_max = int(chin_pt[1])
        cv2.line(draw_img, (mid_x, y_min), (mid_x, y_max), (255, 0, 0), 1)

        total_face_h = chin_pt[1] - fore_top_pt[1]
        line_y1 = fore_top_pt[1] + total_face_h * (1/3)
        line_y2 = fore_top_pt[1] + total_face_h * (2/3)
        cv2.line(draw_img, (0, int(line_y1)), (w, int(line_y1)), (0, 165, 255), 1)
        cv2.line(draw_img, (0, int(line_y2)), (w, int(line_y2)), (0, 165, 255), 1)

        single_eye_w = w / 5
        for i in range(1, 5):
            x_pos = int(single_eye_w * i)
            cv2.line(draw_img, (x_pos, 0), (x_pos, h), (200, 0, 200), 1)

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
            res4 = analyzer.calc_eye_nose_lip()
            res5 = analyzer.calc_face_shape()
            print("三庭五眼：", res1)
            print("面部对称：", res2)
            print("皮肤检测：", res3)
            print("五官比例数据：", res4)
            print("脸型分析：", res5)
            show_img = analyzer.draw_all_landmark()
            cv2.imshow("医美人脸网格分析", show_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()