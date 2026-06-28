import os
import time
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
# 屏蔽MediaPipe底层冗余日志
os.environ['GLOG_minloglevel'] = '2'

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fpdf import FPDF

# ===================== 1、图片保存：解决中文文件名乱码 =====================
def save_cv_image_chinese(file_path: str, cv_img):
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    try:
        pil_img.save(file_path)
        return True
    except Exception as e:
        print(f"图片保存失败：{e}")
        return False

# ===================== 2、图像绘制中文：解决画面文字乱码核心函数 =====================
def draw_chinese_text(cv_img, text_list, start_y, line_gap=26, font_size=14, white_stroke=True):
    """
    cv_img: opencv BGR图像
    text_list: 文本行列表
    start_y: 起始Y坐标
    返回绘制完成后的BGR图像
    """
    h, w = cv_img.shape[:2]
    # OpenCV BGR → PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # 自动匹配系统中文字体
    font = None
    font_paths = []
    if os.name == "nt":
        # Windows
        font_paths = [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]
    else:
        # Mac / Linux
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        ]
    for fp in font_paths:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, font_size)
            break
    # 兜底无中文字体
    if font is None:
        font = ImageFont.load_default()

    y = start_y
    for line in text_list:
        # 白色描边防遮挡
        if white_stroke:
            for offset_x in [-1, 0, 1]:
                for offset_y in [-1, 0, 1]:
                    draw.text((20 + offset_x, y + offset_y), line, font=font, fill=(0, 0, 0))
        # 主文字白色
        draw.text((20, y), line, font=font, fill=(255, 255, 255))
        y += line_gap

    # PIL RGB 转回 OpenCV BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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
    "left_jaw": 58, "right_jaw": 291,
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
        jaw_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                   176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        contour = self.landmarks_2d[jaw_idx].astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)
        skin_bgr = cv2.bitwise_and(self.img, self.img, mask=mask)
        skin_hsv = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(skin_bgr, cv2.COLOR_BGR2GRAY)
        total_skin = np.count_nonzero(mask)
        if total_skin == 0:
            return {
                "泛红面积占比": 0.0, "暗沉面积占比": 0.0,
                "黄褐斑占比": 0.0, "晒斑占比": 0.0,
                "痘印色素沉淀占比": 0.0,
                "毛孔粗糙占比": 0.0, "出油高光占比": 0.0, "干纹缺水占比": 0.0
            }

        red_mask = cv2.inRange(skin_hsv, (0, 80, 50), (20, 255, 255))
        red_pix = np.count_nonzero(red_mask & mask)
        red_ratio = round(float(red_pix / total_skin), 3)

        dark_mask = cv2.inRange(skin_hsv, (0, 0, 0), (180, 255, 90))
        dark_pix = np.count_nonzero(dark_mask & mask)
        dark_ratio = round(float(dark_pix / total_skin), 3)

        sun_low = np.array([10, 60, 60])
        sun_high = np.array([30, 180, 255])
        sun_mask = cv2.inRange(skin_hsv, sun_low, sun_high) & mask

        mel_low = np.array([15, 20, 30])
        mel_high = np.array([35, 70, 120])
        mel_mask = cv2.inRange(skin_hsv, mel_low, mel_high) & mask
        mel_mask = cv2.bitwise_xor(mel_mask, sun_mask)

        acne_mark_low = np.array([0, 30, 40])
        acne_mark_high = np.array([25, 120, 140])
        acne_mask = cv2.inRange(skin_hsv, acne_mark_low, acne_mark_high) & mask
        acne_mask = cv2.bitwise_xor(acne_mask, red_mask)
        acne_mask = cv2.bitwise_xor(acne_mask, mel_mask)
        acne_mask = cv2.bitwise_xor(acne_mask, sun_mask)

        sun_pix = np.count_nonzero(sun_mask)
        mel_pix = np.count_nonzero(mel_mask)
        acne_pix = np.count_nonzero(acne_mask)
        sun_ratio = round(float(sun_pix / total_skin), 3)
        mel_ratio = round(float(mel_pix / total_skin), 3)
        acne_ratio = round(float(acne_pix / total_skin), 3)

        oil_mask = cv2.inRange(skin_hsv, (0, 0, 210), (180, 40, 255))
        oil_pix = np.count_nonzero(oil_mask & mask)
        oil_ratio = round(float(oil_pix / total_skin), 3)

        dry_line_mask = cv2.inRange(skin_hsv, (0, 0, 60), (180, 70, 130))
        dry_pix = np.count_nonzero(dry_line_mask & mask)
        dry_ratio = round(float(dry_pix / total_skin), 3)

        blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        pore_mask = cv2.subtract(blur_gray, gray)
        _, pore_bin = cv2.threshold(pore_mask, 12, 255, cv2.THRESH_BINARY)
        pore_pix = np.count_nonzero(pore_bin & mask)
        pore_ratio = round(float(pore_pix / total_skin), 3)

        return {
            "泛红面积占比": red_ratio,
            "暗沉面积占比": dark_ratio,
            "黄褐斑占比": mel_ratio,
            "晒斑占比": sun_ratio,
            "痘印色素沉淀占比": acne_ratio,
            "毛孔粗糙占比": pore_ratio,
            "出油高光占比": oil_ratio,
            "干纹缺水占比": dry_ratio
        }

    def calc_eye_nose_lip(self):
        left_eye_inner = self.get_point(33)
        left_eye_outer = self.get_point(133)
        left_eye_top = self.get_point(145)
        left_eye_bottom = self.get_point(159)

        right_eye_inner = self.get_point(362)
        right_eye_outer = self.get_point(263)
        right_eye_top = self.get_point(374)
        right_eye_bottom = self.get_point(386)

        brow_mid_left = self.get_point(234)
        brow_mid_right = self.get_point(454)
        nose_root = self.get_point(1)
        nose_tip = self.get_point(4)
        nose_left_wing = self.get_point(21)
        nose_right_wing = self.get_point(24)

        jaw_left_angle = self.get_point(58)
        jaw_right_angle = self.get_point(291)
        chin_tip = self.get_point(152)
        mid_nose_bottom = self.get_point(4)

        left_eye_len = np.linalg.norm(left_eye_outer - left_eye_inner)
        right_eye_len = np.linalg.norm(right_eye_outer - right_eye_inner)
        eye_avg_len = (left_eye_len + right_eye_len) / 2

        left_eye_h = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_h = abs(right_eye_top[1] - right_eye_bottom[1])
        eye_avg_h = (left_eye_h + right_eye_h) / 2

        inter_eye_dist = abs(right_eye_inner[0] - left_eye_inner[0])
        left_eye_tilt = left_eye_outer[1] - left_eye_inner[1]
        right_eye_tilt = right_eye_outer[1] - right_eye_inner[1]
        avg_eye_tilt = (left_eye_tilt + right_eye_tilt) / 2

        left_brow_eye_dist = abs(brow_mid_left[1] - left_eye_top[1])
        right_brow_eye_dist = abs(brow_mid_right[1] - right_eye_top[1])
        avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2

        h, w = self.h, self.w
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        mask_eye_puff = np.zeros((h, w), np.uint8)
        left_puff = np.array([brow_mid_left, left_eye_top, left_eye_inner, left_eye_outer], np.int32)
        right_puff = np.array([brow_mid_right, right_eye_top, right_eye_inner, right_eye_outer], np.int32)
        cv2.fillPoly(mask_eye_puff, [left_puff, right_puff], 255)
        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        puff_tex = cv2.absdiff(gray, blur_gray)
        puff_roi = cv2.bitwise_and(puff_tex, puff_tex, mask=mask_eye_puff)
        puff_std = np.std(puff_roi[puff_roi > 0])
        eye_puff_ratio = round(np.clip(puff_std / 35, 0, 1), 3)

        nose_total_vertical = abs(nose_tip[1] - nose_root[1])
        nose_width = np.linalg.norm(nose_right_wing - nose_left_wing)
        bridge_height = abs(nose_tip[1] - nose_root[1])
        tip_size = np.linalg.norm(np.array([nose_left_wing, nose_right_wing]).mean(axis=0) - nose_tip)
        left_nose_foot = self.get_point(21)
        right_nose_foot = self.get_point(24)
        v1 = nose_tip - np.array([(left_nose_foot[0] + right_nose_foot[0]) / 2, (left_nose_foot[1] + right_nose_foot[1]) / 2])
        v2 = self.get_point(0) - np.array([(left_nose_foot[0] + right_nose_foot[0]) / 2, (left_nose_foot[1] + right_nose_foot[1]) / 2])
        dot = np.dot(v1, v2)
        mod1 = np.linalg.norm(v1)
        mod2 = np.linalg.norm(v2)
        nose_lip_angle = np.arccos(np.clip(dot / (mod1 * mod2), -1, 1)) * 180 / np.pi

        lip_left = self.get_point(61)
        lip_right = self.get_point(291)
        lip_top_mid = self.get_point(0)
        lip_bottom_mid = self.get_point(17)
        lip_cupids_bow = self.get_point(13)
        lip_bottom_center = self.get_point(14)
        philtrum_top = self.get_point(1)

        lip_total_len = np.linalg.norm(lip_right - lip_left)
        upper_lip_h = abs(lip_top_mid[1] - lip_cupids_bow[1])
        lower_lip_h = abs(lip_bottom_mid[1] - lip_bottom_center[1])
        philtrum_len = abs(lip_cupids_bow[1] - philtrum_top[1])
        mouth_tilt = lip_right[1] - lip_left[1]

        jaw_width = np.linalg.norm(jaw_right_angle - jaw_left_angle)
        v_jaw_left_up = brow_mid_left - jaw_left_angle
        v_jaw_left_down = chin_tip - jaw_left_angle
        dot_left = np.dot(v_jaw_left_up, v_jaw_left_down)
        mod_lu = np.linalg.norm(v_jaw_left_up)
        mod_ld = np.linalg.norm(v_jaw_left_down)
        angle_left = np.arccos(np.clip(dot_left / (mod_lu * mod_ld), -1, 1)) * 180 / np.pi
        v_jaw_right_up = brow_mid_right - jaw_right_angle
        v_jaw_right_down = chin_tip - jaw_right_angle
        dot_right = np.dot(v_jaw_right_up, v_jaw_right_down)
        mod_ru = np.linalg.norm(v_jaw_right_up)
        mod_rd = np.linalg.norm(v_jaw_right_down)
        angle_right = np.arccos(np.clip(dot_right / (mod_ru * mod_rd), -1, 1)) * 180 / np.pi
        jaw_angle_avg = (angle_left + angle_right) / 2
        chin_len = abs(chin_tip[1] - mid_nose_bottom[1])
        chin_retreat = chin_tip[0] - mid_nose_bottom[0]

        face_w = np.linalg.norm(self.get_point(PTS_IDX["right_cheek"]) - self.get_point(PTS_IDX["left_cheek"]))
        face_h = abs(self.get_point(PTS_IDX["bottom_chin"])[1] - self.get_point(PTS_IDX["top_forehead"])[1])

        return {
            "眼裂长度(像素均值)": round(float(eye_avg_len), 1),
            "单眼眼高(像素均值)": round(float(eye_avg_h), 1),
            "两眼内眼角间距(像素)": round(float(inter_eye_dist), 1),
            "内外眼角高低差(正=吊梢眼/负=下垂眼)": round(float(avg_eye_tilt), 1),
            "眉眼垂直间距(像素均值)": round(float(avg_brow_eye_dist), 1),
            "肿眼泡区域凸起占比(0-1越高越肿)": eye_puff_ratio,
            "平均眼宽(占脸宽)": round(float(eye_avg_len / face_w), 3),
            "眼间距(占脸宽)": round(float(inter_eye_dist / face_w), 3),
            "单眼平均高度(占脸高)": round(float((left_eye_h + right_eye_h) / 2 / face_h), 3),
            "鼻梁高度(像素)": round(float(bridge_height), 1),
            "鼻翼宽度(像素)": round(float(nose_width), 1),
            "鼻总长(鼻根至鼻尖像素)": round(float(nose_total_vertical), 1),
            "鼻头大小指数(像素)": round(float(tip_size), 1),
            "鼻唇角(°，标准90-95°)": round(float(nose_lip_angle), 1),
            "鼻翼宽度(占脸宽)": round(float(nose_width / face_w), 3),
            "鼻长(眉心至鼻尖/占脸高)": round(float(nose_total_vertical / face_h), 3),
            "嘴唇总长度(左右口角像素)": round(float(lip_total_len), 1),
            "上唇厚度(像素)": round(float(upper_lip_h), 1),
            "下唇厚度(像素)": round(float(lower_lip_h), 1),
            "人中长度(像素)": round(float(philtrum_len), 1),
            "口角高低差(像素，正=右口角偏高)": round(float(mouth_tilt), 1),
            "嘴唇宽度(占脸宽)": round(float(lip_total_len / face_w), 3),
            "上唇厚度(占脸高)": round(float(upper_lip_h / face_h), 3),
            "下唇厚度(占脸高)": round(float(lower_lip_h / face_h), 3),
            "人中长度(占脸高)": round(float(philtrum_len / face_h), 3),
            "下颌宽度(左右下颌角像素)": round(float(jaw_width), 1),
            "下颌角平均角度(°，标准120-130°)": round(float(jaw_angle_avg), 1),
            "下巴纵向长度(鼻底到下巴像素)": round(float(chin_len), 1),
            "下巴后缩指数(负=后缩，正=前突)": round(float(chin_retreat), 1),
            "下颌宽度(占脸宽)": round(float(jaw_width / face_w), 3),
            "下巴长度(占脸高)": round(float(chin_len / face_h), 3)
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

    def calc_aging_depression(self):
        h, w = self.h, self.w
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        texture = cv2.absdiff(gray, blur_gray)
        mask = np.zeros((h, w), dtype=np.uint8)
        jaw_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        contour = self.landmarks_2d[jaw_idx].astype(np.int32)
        cv2.fillPoly(mask, [contour], 255)

        def get_wrinkle_score(roi_poly):
            r_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(r_mask, [roi_poly], 255)
            roi_tex = cv2.bitwise_and(texture, texture, mask=r_mask & mask)
            tex_std = np.std(roi_tex[roi_tex > 0])
            score = np.clip(tex_std / 32, 0, 0.8)
            return round(float(score), 3)

        left_eye_inner = self.get_point(33)
        left_cheek_up = self.get_point(234)
        right_eye_inner = self.get_point(263)
        right_cheek_up = self.get_point(454)
        left_lacrimal_pts = np.array([left_eye_inner, self.get_point(145), left_cheek_up], np.int32)
        right_lacrimal_pts = np.array([right_eye_inner, self.get_point(374), right_cheek_up], np.int32)
        lacrimal_score = get_wrinkle_score(np.concatenate([left_lacrimal_pts, right_lacrimal_pts]))

        nose_left = self.get_point(234)
        mouth_left = self.get_point(61)
        nose_right = self.get_point(454)
        mouth_right = self.get_point(291)
        left_nasolabial = np.array([nose_left, self.get_point(1), mouth_left], np.int32)
        right_nasolabial = np.array([nose_right, self.get_point(4), mouth_right], np.int32)
        nasolabial_score = get_wrinkle_score(np.concatenate([left_nasolabial, right_nasolabial]))

        brow_mid = self.get_point(PTS_IDX["brow_top"])
        cheek_mid_left = self.get_point(234)
        cheek_mid_right = self.get_point(454)
        chin_mid = self.get_point(PTS_IDX["bottom_chin"])
        face_top = self.get_point(PTS_IDX["top_forehead"])
        full_h = chin_mid[1] - face_top[1]
        brow_offset = abs(brow_mid[1] - (face_top[1] + full_h * 0.12)) / full_h
        cheek_left_offset = abs(cheek_mid_left[1] - (face_top[1] + full_h * 0.42)) / full_h
        cheek_right_offset = abs(cheek_mid_right[1] - (face_top[1] + full_h * 0.42)) / full_h
        avg_cheek_offset = (cheek_left_offset + cheek_right_offset) / 2
        sag_score = round(float((brow_offset + avg_cheek_offset) / 2), 3)

        fore_left = self.get_point(234)
        fore_right = self.get_point(454)
        fore_top = self.get_point(10)
        fore_mid = self.get_point(9)
        fore_pts = np.array([fore_left, fore_right, fore_mid, fore_top], np.int32)
        forehead_wrinkle = get_wrinkle_score(fore_pts)

        left_eye_out = self.get_point(133)
        right_eye_out = self.get_point(263)
        left_tail = np.array([left_eye_out, self.get_point(145), self.get_point(159)], np.int32)
        right_tail = np.array([right_eye_out, self.get_point(374), self.get_point(386)], np.int32)
        crow_feet = get_wrinkle_score(np.concatenate([left_tail, right_tail]))

        mouth_left = self.get_point(61)
        mouth_right = self.get_point(291)
        jaw_left = self.get_point(58)
        jaw_right = self.get_point(291)
        left_puppet = np.array([mouth_left, jaw_left, self.get_point(17)], np.int32)
        right_puppet = np.array([mouth_right, jaw_right, self.get_point(17)], np.int32)
        puppet_wrinkle = get_wrinkle_score(np.concatenate([left_puppet, right_puppet]))

        glabala_left = self.get_point(33)
        glabala_right = self.get_point(263)
        glabala_top = self.get_point(9)
        glabala_bottom = self.get_point(1)
        glabala_pts = np.array([glabala_left, glabala_right, glabala_bottom, glabala_top], np.int32)
        glabala_wrinkle = get_wrinkle_score(glabala_pts)

        return {
            "泪沟凹陷程度(0-0.8越高越深)": lacrimal_score,
            "法令纹凹陷程度(0-0.8越深越重)": nasolabial_score,
            "面部软组织下垂指数(0-1越高越松弛)": sag_score,
            "额头横纹深度(0-0.8越深越多)": forehead_wrinkle,
            "鱼尾纹深浅指数(0-0.8越深越多)": crow_feet,
            "木偶纹凹陷程度(0-0.8越深越重)": puppet_wrinkle,
            "眉间川字纹深度(0-0.8越深越多)": glabala_wrinkle
        }

    def calc_total_beauty_score(self):
        shape_data = self.calc_three_court_five_eye()
        sym_data = self.calc_symmetry_score()
        skin_data = self.skin_region_analysis()
        eye_nose_lip = self.calc_eye_nose_lip()
        aging_data = self.calc_aging_depression()

        five_eye = shape_data["五眼匹配度(理想值=1)"]
        sym = sym_data["面部对称得分(满分1)"]
        t1, t2, t3 = shape_data["三庭比例(上/中/下)"]
        three_court_err = abs(t1 - 0.333) + abs(t2 - 0.333) + abs(t3 - 0.333)
        three_court_score = max(0, 1 - three_court_err)
        contour_raw = (five_eye * 0.4 + sym * 0.3 + three_court_score * 0.3)
        contour_score = round(contour_raw * 30, 1)

        eye_w = eye_nose_lip["平均眼宽(占脸宽)"]
        eye_h = eye_nose_lip["单眼平均高度(占脸高)"]
        nose_w = eye_nose_lip["鼻翼宽度(占脸宽)"]
        up_lip = eye_nose_lip["上唇厚度(占脸高)"]
        down_lip = eye_nose_lip["下唇厚度(占脸高)"]

        err_eye = abs(eye_w - 0.085) + abs(eye_h - 0.024)
        err_nose = abs(nose_w - 0.33)
        err_lip = abs((down_lip / up_lip) - 1.3) if up_lip > 0 else 0.25
        total_err = err_eye + err_nose + err_lip
        face_feature_raw = max(0, 1 - total_err / 1.8)
        feature_score = round(face_feature_raw * 30, 1)

        red = skin_data["泛红面积占比"]
        oil = skin_data["出油高光占比"]
        mel = skin_data["黄褐斑占比"]
        sun = skin_data["晒斑占比"]
        acne = skin_data["痘印色素沉淀占比"]
        pore = skin_data["毛孔粗糙占比"]
        dry_line = skin_data["干纹缺水占比"]
        skin_err = red * 0.4 + oil * 0.2 + mel * 0.2 + acne * 0.15 + sun * 0.1 + pore * 0.05 + dry_line * 0.05
        skin_raw = max(0, 1 - skin_err)
        skin_score = round(skin_raw * 20, 1)

        wrinkle_sum = (
            aging_data["泪沟凹陷程度(0-0.8越高越深)"] +
            aging_data["法令纹凹陷程度(0-0.8越深越重)"] +
            aging_data["额头横纹深度(0-0.8越深越多)"] +
            aging_data["鱼尾纹深浅指数(0-0.8越深越多)"] +
            aging_data["木偶纹凹陷程度(0-0.8越深越重)"] +
            aging_data["眉间川字纹深度(0-0.8越深越多)"]
        )
        sag = aging_data["面部软组织下垂指数(0-1越高越松弛)"]
        total_wrinkle = wrinkle_sum * 0.7 + sag * 0.3
        wrinkle_deduct = min(20, round(total_wrinkle * 20, 1))

        total = contour_score + feature_score + skin_score - wrinkle_deduct
        total_clamp = round(max(0, min(100, total)), 1)

        return {
            "轮廓得分(满分30)": contour_score,
            "五官协调得分(满分30)": feature_score,
            "肤质得分(满分20)": skin_score,
            "衰老纹路总扣分项(最高扣20)": wrinkle_deduct,
            "综合颜值总分(0-100)": total_clamp
        }

    def draw_all_landmark(self):
        draw_img = self.img.copy()
        pts_int = self.landmarks_2d.astype(np.int32)

        for x, y in pts_int:
            cv2.circle(draw_img, (x, y), 1, (0, 255, 0), -1)

        jaw_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
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
        line_y1 = fore_top_pt[1] + total_face_h * (1 / 3)
        line_y2 = fore_top_pt[1] + total_face_h * (2 / 3)
        cv2.line(draw_img, (0, int(line_y1)), (w, int(line_y1)), (0, 165, 255), 1)
        cv2.line(draw_img, (0, int(line_y2)), (w, int(line_y2)), (0, 165, 255), 1)

        single_eye_w = w / 5
        for i in range(1, 5):
            x_pos = int(single_eye_w * i)
            cv2.line(draw_img, (x_pos, 0), (x_pos, h), (200, 0, 200), 1)

        return draw_img

    def generate_report_img(self, save_path="report_face.jpg"):
        draw_img = self.draw_all_landmark()
        d1 = self.calc_three_court_five_eye()
        d2 = self.calc_symmetry_score()
        d3 = self.skin_region_analysis()
        d4 = self.calc_eye_nose_lip()
        d5 = self.calc_face_shape()
        d6 = self.calc_aging_depression()
        d7 = self.calc_total_beauty_score()

        text_lines = [
            "==== AI医美面诊报告 ====",
            f"脸型分类：{d5['脸型分类']}",
            f"综合颜值总分：{d7['综合颜值总分(0-100)']} / 100",
            f"轮廓分(30)：{d7['轮廓得分(满分30)']} | 五官分(30)：{d7['五官协调得分(满分30)']}",
            f"肤质分(20)：{d7['肤质得分(满分20)']} | 衰老扣分：{d7['衰老纹路总扣分项(最高扣20)']}",
            "",
            "【肤质重点问题】",
            f"泛红：{d3['泛红面积占比']} 黄褐斑：{d3['黄褐斑占比']} 晒斑：{d3['晒斑占比']}",
            f"痘印色素：{d3['痘印色素沉淀占比']} 出油：{d3['出油高光占比']}",
            "",
            "【核心衰老纹路】",
            f"泪沟：{d6['泪沟凹陷程度(0-0.8越高越深)']} 法令纹：{d6['法令纹凹陷程度(0-0.8越深越重)']}",
            f"鱼尾纹：{d6['鱼尾纹深浅指数(0-0.8越深越多)']} 川字纹：{d6['眉间川字纹深度(0-0.8越深越多)']}"
        ]
        # 替换为PIL绘制中文，彻底删除cv2.putText
        draw_img = draw_chinese_text(draw_img, text_lines, start_y=30, line_gap=26, font_size=14)

        save_path = save_path.replace("\\", "/")
        dir_path = os.path.dirname(save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        success = save_cv_image_chinese(save_path, draw_img)
        if success:
            print(f"[成功] 标注图已保存：{save_path}")
        else:
            print(f"[失败] 标注图保存失败：{save_path}")
        return save_path

    def save_txt_report(self, save_path="report_data.txt"):
        d1 = self.calc_three_court_five_eye()
        d2 = self.calc_symmetry_score()
        d3 = self.skin_region_analysis()
        d4 = self.calc_eye_nose_lip()
        d5 = self.calc_face_shape()
        d6 = self.calc_aging_depression()
        d7 = self.calc_total_beauty_score()

        content = []
        content.append("=" * 60)
        content.append("                AI医美标准化面诊报告")
        content.append("=" * 60)
        content.append(f"脸型类型：{d5['脸型分类']}")
        content.append(f"综合颜值总分：{d7['综合颜值总分(0-100)']} / 100")
        content.append(f"轮廓分项得分(满分30)：{d7['轮廓得分(满分30)']}")
        content.append(f"五官协调得分(满分30)：{d7['五官协调得分(满分30)']}")
        content.append(f"肤质得分(满分20)：{d7['肤质得分(满分20)']}")
        content.append(f"衰老纹路扣分值：{d7['衰老纹路总扣分项(最高扣20)']}")
        content.append("\n【一、轮廓骨骼分析】")
        content.append(f"三庭比例(上/中/下)：{d1['三庭比例(上/中/下)']}")
        content.append(f"五眼匹配标准度：{d1['五眼匹配度(理想值=1)']}")
        content.append(f"面部对称系数：{d2['面部对称得分(满分1)']}")
        content.append(f"脸型比例 脸长/颧骨宽：{d5['脸长/颧骨宽比值']} 下颌/颧骨宽：{d5['下颌宽/颧骨宽比值']}")
        content.append("\n【二、眼部量化数据】")
        for k, v in d4.items():
            if "眼" in k:
                content.append(f"{k}：{v}")
        content.append("\n【三、鼻部量化数据】")
        for k, v in d4.items():
            if "鼻" in k:
                content.append(f"{k}：{v}")
        content.append("\n【四、唇部量化数据】")
        for k, v in d4.items():
            if "唇" in k or "人中" in k:
                content.append(f"{k}：{v}")
        content.append("\n【五、下颌骨骼量化】")
        for k, v in d4.items():
            if "下颌" in k or "下巴" in k:
                content.append(f"{k}：{v}")
        content.append("\n【六、皮肤8项检测】")
        for k, v in d3.items():
            content.append(f"{k}：{v}")
        content.append("\n【七、全脸7处衰老纹路】")
        for k, v in d6.items():
            content.append(f"{k}：{v}")
        content.append("\n" + "=" * 60)
        content.append("改善建议总结：")
        tips = []
        if d7["综合颜值总分(0-100)"] < 60:
            tips.append("整体多项基础条件偏差，建议联合轮廓+五官+皮肤综合改善")
        if d3["泛红面积占比"] > 0.15:
            tips.append("皮肤泛红敏感，优先舒敏光子修护屏障")
        if d3["黄褐斑占比"] > 0.05:
            tips.append("存在黄褐斑，温和代谢、内调，禁用强爆破激光")
        if d3["晒斑占比"] > 0.05:
            tips.append("表层晒斑，皮秒/超皮秒爆破+严格防晒")
        if d3["痘印色素沉淀占比"] > 0.05:
            tips.append("炎症痘印，果酸焕肤+烟酰胺淡化色素")
        if d6["鱼尾纹深浅指数(0-0.8越深越多)"] > 0.3:
            tips.append("鱼尾纹动态纹，肉毒素除皱")
        if d6["泪沟凹陷程度(0-0.8越高越深)"] > 0.3:
            tips.append("泪沟凹陷，眶下玻尿酸填充")
        if d4["鼻翼宽度(占脸宽)"] > 0.34:
            tips.append("鼻翼宽大，建议鼻综合缩鼻翼")
        if d4["单眼平均高度(占脸高)"] < 0.022:
            tips.append("眼高不足，双眼皮+提肌放大双眼")
        if len(tips) == 0:
            tips.append("基础条件良好，仅日常轻护理维持即可")
        content.extend(tips)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        return save_path

    def save_pdf_report(self, save_path="report.pdf"):
        pdf = FPDF()
        pdf.add_page()
        font_ok = False
        font_paths = []
        if os.name == "nt":
            font_paths = [
                ("msyh", r"C:\Windows\Fonts\msyh.ttc"),
                ("simhei", r"C:\Windows\Fonts\simhei.ttf")
            ]
        else:
            font_paths = [
                ("wqy", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
                ("pingfang", "/System/Library/Fonts/PingFang.ttc"),
                ("stheitisc", "/Library/Fonts/STHeitiMedium.ttc")
            ]
        for font_name, font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    pdf.add_font(font_name, "", font_path)
                    pdf.set_font(font_name, size=11)
                    font_ok = True
                    break
            except:
                continue
        if not font_ok:
            pdf.set_font("DejaVuSans", size=11)

        tmp_txt = "tmp_report.txt"
        self.save_txt_report(tmp_txt)
        with open(tmp_txt, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            pdf.multi_cell(190, 6, text=line.strip())
        pdf.output(save_path)
        if os.path.exists(tmp_txt):
            os.remove(tmp_txt)
        return save_path

    def save_report_all(self, out_dir="./report_out"):
        stamp = str(int(time.time()))
        out_dir = out_dir.replace("\\", "/")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        img_path = os.path.join(out_dir, f"面诊标注图_{stamp}.jpg")
        txt_path = os.path.join(out_dir, f"面诊文字报告_{stamp}.txt")
        pdf_path = os.path.join(out_dir, f"面诊标准化报告_{stamp}.pdf")
        p1 = self.generate_report_img(img_path)
        p2 = self.save_txt_report(txt_path)
        p3 = self.save_pdf_report(pdf_path)
        print(f"\n报告输出目录：{os.path.abspath(out_dir)}")
        print(f"1、人脸标注图：{p1}")
        print(f"2、文字TXT报告：{p2}")
        print(f"3、打印PDF报告：{p3}")
        return {"img": p1, "txt": p2, "pdf": p3}

if __name__ == "__main__":
    # 安装依赖
    # pip install opencv-python mediapipe fpdf pillow numpy
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
            res6 = analyzer.calc_aging_depression()
            res7 = analyzer.calc_total_beauty_score()
            print("三庭五眼：", res1)
            print("面部对称：", res2)
            print("皮肤检测：", res3)
            print("五官比例数据：", res4)
            print("脸型分析：", res5)
            print("衰老松弛纹路检测：", res6)
            print("综合颜值打分：", res7)

            report_paths = analyzer.save_report_all()

            show_img = analyzer.draw_all_landmark()
            cv2.imshow("医美人脸网格分析", show_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()