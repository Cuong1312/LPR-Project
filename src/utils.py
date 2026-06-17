import os, time, threading, collections, re
import hashlib, io
import logging
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv() # Khởi tạo môi trường

# Khởi tạo mô hình nhận diện ký tự bằng YOLO cục bộ
print("Đang tải mô hình YOLO Character...")
try:
    char_model = YOLO('models/yolov8n_char.pt') 
    print("Tải YOLO Character thành công ✓")
except Exception as e:
    char_model = None
    print(f"[PlateVision] Lỗi tải YOLO Character: {e}")

_ocr_cache  = {}
_cache_lock = threading.Lock()

def _crop_hash(pil_img):
    thumb = pil_img.convert("L").resize((32, 16))
    return hashlib.md5(thumb.tobytes()).hexdigest()

def _sharpness_score(pil_img) -> float:
    """
    Đo độ nét của ảnh crop bằng Laplacian variance.
    Số càng cao = ảnh càng nét = biển số đang ở khoảng cách vừa tầm.
    Dùng để chọn frame tốt nhất trong lịch sử track.
    """
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    # Upscale nhỏ nếu quá nhỏ để Laplacian có đủ pixel để đánh giá
    h, w = gray.shape
    if w < 80:
        gray = cv2.resize(gray, (80, int(80*h/w)))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def preprocess_plate(img_cv):
    img_cv = cv2.resize(img_cv, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return blurred

def ocr_plate(pil_img, track_id=-1):
    h = _crop_hash(pil_img)
    with _cache_lock:
        if h in _ocr_cache: return _ocr_cache[h]

    # Luồng 1: Chạy YOLO trực tiếp
    result = yolo_ocr_fallback(pil_img)

    # Luồng 2: Nếu không đọc được, tiền xử lý ảnh và cho YOLO chạy lại
    if result == "[Không đọc được]" or len(result) < 5:
        img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        processed_cv = preprocess_plate(img_cv)
        processed_pil = Image.fromarray(cv2.cvtColor(processed_cv, cv2.COLOR_GRAY2RGB))
        result = yolo_ocr_fallback(processed_pil)

    with _cache_lock:
        _ocr_cache[h] = result
        if len(_ocr_cache) > 500: del _ocr_cache[next(iter(_ocr_cache))]
    return result

import re

def format_vietnamese_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(text) < 7:
        return text
    chars = list(text)
    num_to_char = {'8': 'B', '0': 'D', '5': 'S', '2': 'Z', '1': 'I', '4': 'A'}
    char_to_num = {'B': '8', 'D': '0', 'O': '0', 'S': '5', 'Z': '2', 'I': '1', 'G': '6', 'A': '4'}

    for i in range(2):
        if chars[i] in char_to_num: chars[i] = char_to_num[chars[i]]
    if chars[2] in num_to_char: chars[2] = num_to_char[chars[2]]
    for i in range(len(chars) - 5, len(chars)):
        if i >= 3 and chars[i] in char_to_num: chars[i] = char_to_num[chars[i]]

    formatted_text = "".join(chars)
    if len(formatted_text) == 8: return f"{formatted_text[:3]}-{formatted_text[3:6]}.{formatted_text[6:]}"
    elif len(formatted_text) == 9: return f"{formatted_text[:4]}-{formatted_text[4:7]}.{formatted_text[7:]}"
    elif len(formatted_text) == 7: return f"{formatted_text[:3]}-{formatted_text[3:]}"
    return formatted_text

def yolo_ocr_fallback(pil_img):
    if not char_model:
        return "[Cần tải mô hình yolov8n_char.pt]"
        
    img_cv = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    
    results = char_model(img_cv, verbose=False, conf=0.25)
    
    if len(results[0].boxes) == 0:
        return "[Không đọc được]"
        
    boxes = results[0].boxes
    chars = []
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        char_text = char_model.names[cls_id]
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        chars.append({
            'char': char_text.upper(),
            'cx': center_x,
            'cy': center_y
        })
        
    w, h = pil_img.size
    aspect_ratio = w / h  # Tính toán tỷ lệ khung hình
    
    # Ranh giới phân chia: Tỷ lệ chiều dài / chiều cao > 2.5 chắc chắn là biển 1 dòng
    if aspect_ratio > 2.5:
        # Xử lý biển dài (1 dòng): Chỉ cần sắp xếp theo trục X từ trái qua phải
        chars = sorted(chars, key=lambda k: k['cx'])
        raw_text = "".join([c['char'] for c in chars])
    else:
        # Xử lý biển vuông (2 dòng): Áp dụng thuật toán cắt trục Y
        avg_y = sum(c['cy'] for c in chars) / len(chars)
        
        line_1 = []
        line_2 = []
        
        for c in chars:
            if c['cy'] < avg_y:
                line_1.append(c)
            else:
                line_2.append(c)
                
        line_1 = sorted(line_1, key=lambda k: k['cx'])
        line_2 = sorted(line_2, key=lambda k: k['cx'])
        
        str_line_1 = "".join([c['char'] for c in line_1])
        str_line_2 = "".join([c['char'] for c in line_2])
        raw_text = str_line_1 + str_line_2
    # --------------------------------------------------
    
    final_text = format_vietnamese_plate(raw_text)
    
    return final_text

# ═══════════════════════════════════════════════
#  IOU Tracker — velocity prediction + per-frame smooth
# ═══════════════════════════════════════════════
def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = max(1, (a[2]-a[0])*(a[3]-a[1]))
    ab = max(1, (b[2]-b[0])*(b[3]-b[1]))
    return inter / (aa + ab - inter)


class Track:
    """
    Theo dõi một biển số với:
    - velocity: tốc độ di chuyển trung bình (pixel/giây)
    - smooth_bbox: được cập nhật mỗi 30ms trên main thread
    """
    _next_id = 1
    COLORS = [
        (0, 229, 255),   # cyan
        (57, 255, 20),   # green neon
        (255, 165, 0),   # orange
        (255, 50, 150),  # pink
        (150, 100, 255), # purple
        (0, 200, 100),   # teal
    ]

    # Số frame tối thiểu trước khi gửi OCR
    MIN_AGE_FOR_OCR = 1   # OCR từ lần detect đầu (ảnh tĩnh chỉ có 1 lần)
    POOL_SIZE       = 5   # giữ top-5 crop sắc nét nhất

    def __init__(self, bbox, text, conf):
        self.id         = Track._next_id
        Track._next_id += 1
        b = [float(x) for x in bbox]
        self.smooth     = b[:]
        self.target     = b[:]
        self.prev_target= b[:]
        self.velocity   = [0.0, 0.0, 0.0, 0.0]
        self.text       = text
        self.conf       = conf
        self.missed     = 0
        self.age        = 1   # tính lần detect đầu tiên là age=1
        self.last_yolo_t= time.time()
        self.color      = self.COLORS[self.id % len(self.COLORS)]

        # Pool ảnh crop tốt nhất: list of (sharpness, pil_img, yolo_conf)
        self.crop_pool  = []
        # Trạng thái OCR: "pending" | "sent" | "done"
        self.ocr_state  = "pending"
        self.best_text  = ""

    def yolo_update(self, bbox, text, conf, pil_crop=None):
        """Được gọi từ YOLO thread khi có detection mới."""
        now = time.time()
        dt  = max(0.05, now - self.last_yolo_t)
        b   = [float(x) for x in bbox]
        for i in range(4):
            self.velocity[i] = (b[i] - self.target[i]) / dt
        self.prev_target  = self.target[:]
        self.target       = b[:]
        self.last_yolo_t  = now
        self.missed       = 0
        self.age         += 1
        if conf > self.conf:
            self.conf = conf

        # Thêm crop vào pool nếu còn chỗ hoặc nét hơn crop tệ nhất
        if pil_crop is not None and self.ocr_state != "done":
            score = _sharpness_score(pil_crop)
            bw = bbox[2] - bbox[0]  # bonus cho biển gần (rộng hơn)
            score += bw * 0.5       # ưu tiên biển to/gần hơn
            if len(self.crop_pool) < self.POOL_SIZE:
                self.crop_pool.append((score, pil_crop, conf))
            else:
                # Thay thế crop tệ nhất nếu crop mới tốt hơn
                worst_i = min(range(len(self.crop_pool)),
                              key=lambda i: self.crop_pool[i][0])
                if score > self.crop_pool[worst_i][0]:
                    self.crop_pool[worst_i] = (score, pil_crop, conf)

    def best_crop(self):
        """Trả về crop sắc nét nhất trong pool."""
        if not self.crop_pool:
            return None, 0.0
        best = max(self.crop_pool, key=lambda x: x[0])
        return best[1], best[2]  # (pil_img, conf)

    def ready_for_ocr(self) -> bool:
        if self.ocr_state != "pending" or len(self.crop_pool) == 0:
            return False

        # Đo lường chiều rộng (Width) của Bounding Box để ước tính khoảng cách
        box_width = self.target[2] - self.target[0]

        # Điều kiện 1: Biển số đã tiến đến đủ gần (Kích thước Bounding Box >= 60 pixel)
        is_close_enough = box_width >= 60

        # Điều kiện 2: Đã bám vết và gom đủ số lượng khung hình (Age >= 12)
        # Hệ thống sẽ tự động đối chiếu hàm _sharpness_score để bốc ra ảnh nét nhất
        is_mature = self.age >= 12

        # Điều kiện 3: Ngắt kết nối (Missed). Khi xe bị che khuất hoặc vọt ra khỏi tầm nhìn camera
        # -> Ép buộc thực thi OCR khẩn cấp với dữ liệu tốt nhất đang có trong Pool
        urgent = self.missed >= 1

        return is_close_enough or is_mature or urgent

    def step(self, dt=0.03):
        ALPHA = 0.85 
        if self.missed == 0:
            # Bám sát theo tọa độ thật của YOLO
            for i in range(4):
                self.smooth[i] += (self.target[i] - self.smooth[i]) * ALPHA

    def opacity(self):
        """0.0 → 1.0, mờ dần khi missed tăng."""
        return max(0.15, 1.0 - self.missed * 0.12)

    def as_draw(self):
        return (int(self.smooth[0]), int(self.smooth[1]),
                int(self.smooth[2]), int(self.smooth[3]),
                self.conf, self.text, self.color, self.id,
                self.opacity(), self.ocr_state)


class IOUTracker:
    MAX_MISSED = 5
    MIN_IOU    = 0.05

    def __init__(self):
        self.tracks: list[Track] = []

    def yolo_update(self, detections):
        """Gọi từ YOLO thread. detections = [(x1,y1,x2,y2,conf,text),...]"""
        self.yolo_update_with_crops(
            [(d[0],d[1],d[2],d[3],d[4],d[5],None) for d in detections]
        )

    def yolo_update_with_crops(self, detections):
        matched_det = set()
        matched_trk = set()

        for di, det in enumerate(detections):
            best_iou, best_ti = 0, -1
            for ti, trk in enumerate(self.tracks):
                iou = _iou(det[:4], trk.target)
                if iou > best_iou:
                    best_iou, best_ti = iou, ti
            if best_iou >= self.MIN_IOU:
                pil_crop = det[6] if len(det) > 6 else None
                self.tracks[best_ti].yolo_update(
                    det[:4], det[5], det[4], pil_crop=pil_crop)
                matched_det.add(di); matched_trk.add(best_ti)

        for ti, trk in enumerate(self.tracks):
            if ti not in matched_trk:
                trk.missed += 1

        for di, det in enumerate(detections):
            if di not in matched_det:
                pil_crop = det[6] if len(det) > 6 else None
                new_trk  = Track(det[:4], det[5], det[4])
                if pil_crop:
                    score = _sharpness_score(pil_crop) + (det[2]-det[0]) * 0.5
                    new_trk.crop_pool.append((score, pil_crop, det[4]))
                self.tracks.append(new_trk)

        evicted = [t for t in self.tracks
                   if t.missed > self.MAX_MISSED
                   and t.ocr_state in ("pending", "sent")
                   and len(t.crop_pool) > 0]
        self.tracks = [t for t in self.tracks if t.missed <= self.MAX_MISSED]
        return evicted

    def step(self, dt=0.03):
        """Gọi mỗi 30ms trên main thread → smooth chuyển động."""
        for trk in self.tracks:
            trk.step(dt)
        return [t for t in self.tracks if t.missed <= self.MAX_MISSED]
    def clear(self):
        self.tracks.clear()
        Track._next_id = 1


# ═══════════════════════════════════════════════
#  Draw tracks lên frame (dùng interpolated bbox)
# ═══════════════════════════════════════════════
# ═══════════════════════════════════════════════
#  Draw tracks lên frame (dùng interpolated bbox)
# ═══════════════════════════════════════════════
def draw_tracks(frame: np.ndarray, tracks: list) -> np.ndarray:
    out = frame.copy()
    for trk in tracks:
        if trk.missed > 0:
            continue
        x1, y1, x2, y2, conf, text, color, tid, opacity, ocr_state = trk.as_draw()

        c       = (0, 140, 255)   # Mã màu cam (hệ BGR)
        c_solid = (0, 140, 255)   # Nền nhãn cũng màu cam
        thickness = 2

        # Vẽ Bounding box (Khung chữ nhật)
        cv2.rectangle(out, (x1, y1), (x2, y2), c, thickness)

        # Vẽ Corner brackets (Góc bo 4 cạnh cho đẹp)
        cs = 14
        for cx, cy, dx, dy in [
            (x1,y1, 1, 1),(x2,y1,-1, 1),
            (x1,y2, 1,-1),(x2,y2,-1,-1)
        ]:
            cv2.line(out, (cx, cy), (cx+dx*cs, cy), c, 2)
            cv2.line(out, (cx, cy), (cx, cy+dy*cs), c, 2)

        if ocr_state == "done" and text and text != "[Không đọc được]":
            # Đã nhận diện được: Hiện thẳng biển số
            label = f"{text}" 
        else:
            # Chưa nhận diện xong: Chỉ hiện thông báo đang quét
            label = f"..."

        # Vẽ phần nhãn thông tin
        fs, th2 = 0.55, 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th2)
        lx = x1; ly = max(th + 10, y1)
        
        # Nền màu cam, chữ màu trắng tinh
        cv2.rectangle(out, (lx, ly-th-8), (lx+tw+8, ly), c_solid, -1)
        txt_color = (255, 255, 255) 
        cv2.putText(out, label, (lx+4, ly-4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, txt_color, th2)
    return out