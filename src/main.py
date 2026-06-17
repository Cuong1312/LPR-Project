import os, threading, queue, time, re
import concurrent.futures
import logging
import cv2
import numpy as np
import pandas as pd
import collections
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


from utils import ocr_plate, IOUTracker, draw_tracks, _crop_hash

# Kích hoạt môi trường 
load_dotenv()

# KHỞI TẠO MÔ HÌNH:
print("Đang tải mô hình YOLO...")
model_yolo = YOLO('models/yolov8n_best.pt')
print("Tải YOLO thành công ✓")

# ═══════════════════════════════════════════════
#  Main Application
# ═══════════════════════════════════════════════
class LRPApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("App")
        self.geometry("1350x780")
        self.minsize(1100, 680)
        

        # Queues
        self._q_display = queue.Queue(maxsize=2)
        self._q_yolo    = queue.Queue(maxsize=1)
        self._q_result  = queue.Queue(maxsize=64)
        self.ocr_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # State
        self.cap             = None
        self._reader_running = False
        self._yolo_running   = False
        self._current_source = None
        self._current_img    = None
        self._last_frame_bgr = None
        self._photo_refs     = collections.OrderedDict()
        self.MAX_PHOTO_CACHE = 50

        # Tracker (updated on main thread via _poll_display)
        self._tracker      = IOUTracker()
        self._tracks_lock  = threading.Lock()
        self._active_tracks: list = []

        self.detected_plates = []
        self.selected_plate  = None

        self.recent_plates = {}    
        self.PLATE_TIMEOUT = 5.0  

        self._build_ui()

        self._yolo_running = True
        threading.Thread(target=self._yolo_worker, daemon=True).start()
        self._poll_display()
        self._poll_results()

    # ═══════════════════════════════════════════════
    #  UI
    # ═══════════════════════════════════════════════
    def _build_ui(self):
        ctk.set_appearance_mode("light")
        self.configure(fg_color="#F0F4F8") 
        
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self._build_toolbar()
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()

    def _build_toolbar(self):
        bar = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color="#FFFFFF")
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            bar, text="App",
            font=ctk.CTkFont("Segoe UI", 22, "bold"), 
            text_color="#2563EB" 
        ).grid(row=0, column=0, padx=20, pady=12)

        bf = ctk.CTkFrame(bar, fg_color="transparent")
        bf.grid(row=0, column=1)
        
        # Bảng màu 
        for txt, cmd, fg, hov, tc in [
            ("📁 Tải Dữ Liệu", self._open_file,      "#F1F5F9", "#E2E8F0", "#1E293B"),
            ("▶ Bắt đầu", self._start_detection, "#2563EB", "#1D4ED8", "#FFFFFF"),
            ("⏹ Tạm Dừng",    self._stop,            "#FEE2E2", "#FECACA", "#991B1B"),
        ]:
            ctk.CTkButton(bf, text=txt, command=cmd, text_color=tc,
                          width=135, height=38, corner_radius=8, fg_color=fg, hover_color=hov,
                          font=ctk.CTkFont("Segoe UI", 13, "bold")).pack(side="left", padx=6)

        self.lbl_count = ctk.CTkLabel(
            bar, text="Phát hiện: 0",
            font=ctk.CTkFont("Segoe UI", 14, "bold"), text_color="#64748B"
        )
        
        self.lbl_count.grid(row=0, column=2, padx=20)

    def _build_left_panel(self):
        panel = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color="#FFFFFF")
        panel.grid(row=1, column=0, sticky="nsew", padx=(0, 2))
        panel.grid_propagate(False)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(panel, fg_color="#F8FAFC", corner_radius=0, height=50)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="LỊCH SỬ QUÉT",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color="#0F172A").place(relx=0.5, rely=0.5, anchor="center")

        self.plates_scroll = ctk.CTkScrollableFrame(
            panel, fg_color="#FFFFFF", scrollbar_button_color="#CBD5E1"
        )
        self.plates_scroll.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self.plates_scroll.grid_columnconfigure(0, weight=1)

        self.lbl_empty = ctk.CTkLabel(
            self.plates_scroll,
            text="Chưa có dữ liệu\n✨ Đang chờ tín hiệu ✨",
            font=ctk.CTkFont("Segoe UI", 13), text_color="#94A3B8", justify="center"
        )
        self.lbl_empty.grid(row=0, column=0, pady=50)

    def _build_center_panel(self):
        panel = ctk.CTkFrame(self, corner_radius=16, fg_color="#FFFFFF")
        panel.grid(row=1, column=1, sticky="nsew", padx=12, pady=12)
        panel.grid_rowconfigure(0, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        self.canvas = ctk.CTkCanvas(panel, bg="#E2E8F0", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 0))

        # --- BỘ ĐIỀU KHIỂN VIDEO ---
        control_frame = ctk.CTkFrame(panel, fg_color="transparent")
        control_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=8)
        control_frame.grid_columnconfigure(1, weight=1)

        self.btn_pause = ctk.CTkButton(control_frame, text="⏸ Tạm Dừng", width=100,
                                       command=self._toggle_pause, font=ctk.CTkFont("Segoe UI", 12, "bold"),
                                       fg_color="#F1F5F9", text_color="#1E293B", hover_color="#E2E8F0")
        self.btn_pause.grid(row=0, column=0, padx=(0, 10))

        self.video_slider = ctk.CTkSlider(control_frame, from_=0, to=100, command=self._seek_video)
        self.video_slider.grid(row=0, column=1, sticky="ew")
        self.video_slider.set(0)
        self.is_paused = False
        # --------------------------------------

        self.lbl_status = ctk.CTkLabel(
            panel, text="Sẵn sàng nhận diện. Vui lòng cấp nguồn video.",
            font=ctk.CTkFont("Segoe UI", 13, "bold"), text_color="#3B82F6"
        )
        self.lbl_status.grid(row=2, column=0, pady=(0, 8)) 
        self.canvas.bind("<Configure>", lambda e: self._redraw_last())

    def _build_right_panel(self):
        panel = ctk.CTkFrame(self, width=290, corner_radius=0, fg_color="#FFFFFF")
        panel.grid(row=1, column=2, sticky="nsew", padx=(2, 0))
        panel.grid_propagate(False)
        panel.grid_rowconfigure(1, weight=1)
        panel.grid_columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(panel, fg_color="#F8FAFC", corner_radius=0, height=50)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_propagate(False)
        ctk.CTkLabel(hdr, text="BÁO CÁO CHI TIẾT",
                     font=ctk.CTkFont("Segoe UI", 12, "bold"),
                     text_color="#0F172A").place(relx=0.5, rely=0.5, anchor="center")

        self.detail_scroll = ctk.CTkScrollableFrame(
            panel, fg_color="#FFFFFF", scrollbar_button_color="#CBD5E1"
        )
        self.detail_scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.detail_scroll.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self.detail_scroll,
                     text="← Chọn một thẻ\nđể xem báo cáo",
                     font=ctk.CTkFont("Segoe UI", 13),
                     text_color="#94A3B8", justify="center"
                     ).grid(row=0, column=0, pady=50)

        self.btn_download = ctk.CTkButton(
            panel, text="⬇ Lưu File Dữ Liệu",
            command=self._download_crop,
            fg_color="#F1F5F9", hover_color="#E2E8F0", text_color="#1E293B",
            font=ctk.CTkFont("Segoe UI", 13, "bold"), corner_radius=8, height=40
        )

    # ═══════════════════════════════════════════════
    #  PLATE CARD
    # ═══════════════════════════════════════════════
    def _add_plate_card(self, det: dict):
        self.lbl_empty.grid_remove()

        # LỆNH TỰ ĐỘNG XÓA ẢNH CŨ 
        if len(self._photo_refs) >= self.MAX_PHOTO_CACHE:
            oldest_key = next(iter(self._photo_refs))
            del self._photo_refs[oldest_key]
        

        # Đẩy cards cũ xuống
        for child in self.plates_scroll.winfo_children():
            info = child.grid_info()
            if info:
                child.grid(row=int(info["row"]) + 1, column=0,
                           sticky="ew", padx=4, pady=3)

        # Thẻ biển số nền trắng, viền xám nhạt, bo tròn
        card = ctk.CTkFrame(self.plates_scroll, fg_color="#FFFFFF",
                            corner_radius=12, border_width=1,
                            border_color="#E2E8F0")
        card.grid(row=0, column=0, sticky="ew", padx=4, pady=3)
        card.grid_columnconfigure(0, weight=1)

        ref_key   = f"thumb_{id(det)}"
        img_frame = None
        img_lbl   = None
        try:
            pil_crop = det["pil_crop"]
            if pil_crop is None or pil_crop.size[0] == 0:
                raise ValueError("empty crop")

            orig_w, orig_h = pil_crop.size
            target_w = 198
            target_h = max(44, int(orig_h * target_w / orig_w))
            pil_resized = pil_crop.resize((target_w, target_h), Image.LANCZOS)

            tk_img = ctk.CTkImage(light_image=pil_resized, size=pil_resized.size)
            self._photo_refs[ref_key] = tk_img

            img_frame = ctk.CTkFrame(card, fg_color="#060a0e",
                                     corner_radius=4, border_width=1,
                                     border_color="#1e4a5a")
            img_frame.grid(row=0, column=0, padx=8, pady=(8, 2), sticky="ew")
            img_lbl = ctk.CTkLabel(img_frame, image=tk_img, text="",
                                   fg_color="transparent")
            img_lbl.pack(padx=3, pady=3)
        except Exception:
            img_frame = ctk.CTkFrame(card, fg_color="#060a0e",
                                     corner_radius=4, border_width=1,
                                     border_color="#1e4a5a", height=44)
            img_frame.grid(row=0, column=0, padx=8, pady=(8, 2), sticky="ew")
            ctk.CTkLabel(img_frame, text="[ no image ]",
                         font=ctk.CTkFont(size=10),
                         text_color="#3a5060").pack(expand=True)

        ctk.CTkLabel(card, text=det["plate_text"],
                     font=ctk.CTkFont("Segoe UI", 16, "bold"),
                     text_color="#2563EB").grid(row=1, column=0, padx=6, pady=(4, 0))

        ctk.CTkLabel(card,
                     text=f"Conf: {det['confidence']:.1%}  ·  {det['time']}",
                     font=ctk.CTkFont(size=10),
                     text_color="#4a90a4").grid(row=2, column=0, padx=6, pady=(0, 8))

        for w in ([card, img_frame] + ([img_lbl] if img_lbl else [])):
            if w:
                w.bind("<Button-1>", lambda e, d=det: self._select_plate(d))

        self.detected_plates.append(det)
        self.lbl_count.configure(text=f"Đã phát hiện: {len(self.detected_plates)}")

    # ═══════════════════════════════════════════════
    #  OPEN FILE / WEBCAM
    # ═══════════════════════════════════════════════
    def _open_file(self):
        self._stop()
        path = filedialog.askopenfilename(
            filetypes=[
                ("Media files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.mp4;*.avi;*.mov;*.mkv"),
                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.webp"),
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv"),
                ("All files", "*.*") 
            ]
        )
        if not path:
            return
        self._current_source = path
        self._current_img    = None
        ext = os.path.splitext(path)[1].lower()

        if ext in (".mp4", ".avi", ".mov", ".mkv"):
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                self._set_status(f"❌ Không mở được: {os.path.basename(path)}")
                self.cap = None; return
            ret, frame = self.cap.read()
            if ret:
                self._current_img = frame
                self._show_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fps   = self.cap.get(cv2.CAP_PROP_FPS)
            total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._set_status(
                f"📹 {os.path.basename(path)}  ({total} frames · {fps:.0f} fps)"
                "  — nhấn ▶ Nhận diện")
        else:
            img = cv2.imread(path)
            if img is None:
                self._set_status("❌ Không đọc được ảnh"); return
            self._current_img = img
            self._show_frame(img)
            self._set_status(f"🖼 {os.path.basename(path)}  — nhấn ▶ Nhận diện")

    def _open_webcam(self):
        self._stop()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._set_status("❌ Không thể mở webcam")
            self.cap = None; return
        self._current_source = "webcam"
        self._set_status("📷 Webcam  — nhấn ▶ Nhận diện")

    # ═══════════════════════════════════════════════
    #  START / STOP
    # ═══════════════════════════════════════════════
    def _start_detection(self):
        if self._current_source is None:
            self._set_status("⚠ Hãy tải ảnh hoặc video trước"); return
        if self._reader_running:
            self._set_status("⚠ Đang chạy rồi — nhấn Dừng trước"); return

        ext = os.path.splitext(self._current_source)[1].lower() \
              if self._current_source != "webcam" else ""
        is_video = ext in (".mp4", ".avi", ".mov", ".mkv") \
                   or self._current_source == "webcam"

        self._tracker.clear()
        self.recent_plates.clear()

        if is_video:
            if self.cap is None or not self.cap.isOpened():
                src = 0 if self._current_source == "webcam" else self._current_source
                self.cap = cv2.VideoCapture(src)
                if not self.cap.isOpened():
                    self._set_status("Không mở được nguồn video"); return
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._reader_running = True
            threading.Thread(target=self._reader_loop, daemon=True).start()
            self._set_status("▶ Đang nhận diện…")
        else:
            if self._current_img is not None:
                self._set_status("⏳ Đang nhận diện ảnh…")
                frame = self._current_img.copy()
        
                self._last_frame_bgr = frame
                try: self._q_display.put_nowait(frame)
                except queue.Full: pass
                # Gửi YOLO — dùng thread riêng để không block UI
                def _run_image_detection(f=frame, src=self._current_source):
                    try: self._q_yolo.put(f, src)
                    except: pass
                try:
                    self._q_yolo.put_nowait((frame, self._current_source))
                except queue.Full:
                    pass

    def _stop(self):
        self._reader_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        for q in (self._q_display, self._q_yolo):
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
        self._tracker.clear()
        with self._tracks_lock:
            self._active_tracks = []
        self._set_status("⏹ Đã dừng.")

    # ═══════════════════════════════════════════════
    #  THREAD 1 — READER
    # ═══════════════════════════════════════════════
    def _toggle_pause(self):
        if self.cap is None: return
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.configure(text="▶ Tiếp Tục")
            self._set_status("⏸ Đang tạm dừng video")
        else:
            self.btn_pause.configure(text="⏸ Tạm Dừng")
            self._set_status("▶ Đang nhận diện…")

    def _seek_video(self, value):
        # UI Thread: Chỉ gửi yêu cầu tua đến luồng đọc video
        self._seek_target = int(value)
    def _reader_loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) if self.cap else 25
        if fps <= 0 or fps > 120: fps = 25
        frame_delay = 1.0 / fps
        yolo_every  = max(1, int(fps / 5))
        
        # Cập nhật độ dài tối đa của thanh tua video
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            self.after(0, lambda: self.video_slider.configure(to=total_frames))

        while self._reader_running:
            t0 = time.time()
            if not self.cap or not self.cap.isOpened(): break

            # --- BƯỚC 1: XỬ LÝ LỆNH TUA TỪ NGƯỜI DÙNG ---
            seek_target = getattr(self, '_seek_target', None)
            if seek_target is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                
                if getattr(self, 'is_paused', False):
                    ret, frame = self.cap.read()
                    if ret:
                        try:
                            while not self._q_display.empty(): self._q_display.get_nowait()
                            self._q_display.put_nowait(frame.copy())
                        except: pass
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, seek_target)
                
                self._seek_target = None 
                continue 

            # --- BƯỚC 2: XỬ LÝ TRẠNG THÁI TẠM DỪNG ---
            if getattr(self, 'is_paused', False):
                time.sleep(0.05) 
                continue

            # --- BƯỚC 3: CHẠY VIDEO BÌNH THƯỜNG ---
            ret, frame = self.cap.read()
            if not ret: break

            frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Cho thanh tua trượt theo video 
            if frame_idx % 5 == 0:
                self.after(0, lambda v=frame_idx: self.video_slider.set(v))

            # Đẩy ảnh lên màn hình hiển thị
            if self._q_display.full():
                try: self._q_display.get_nowait()
                except queue.Empty: pass
            try: self._q_display.put_nowait(frame.copy())
            except queue.Full: pass

            # Đẩy ảnh cho AI nhận diện
            if frame_idx % yolo_every == 0:
                if self._q_yolo.full():
                    try: self._q_yolo.get_nowait()
                    except queue.Empty: pass
                try: self._q_yolo.put_nowait((frame.copy(), self._current_source))
                except queue.Full: pass

            wait = frame_delay - (time.time() - t0)
            if wait > 0: time.sleep(wait)

        self._reader_running = False
        self._set_status("✅ Phân tích hoàn tất")

    # ═══════════════════════════════════════════════
    #  THREAD 2 — YOLO WORKER
    # ═══════════════════════════════════════════════
    def _yolo_worker(self):
        while self._yolo_running:
            try:
                frame, source = self._q_yolo.get(timeout=0.5)
            except queue.Empty:
                continue

            h_f, w_f = frame.shape[:2]

           
            boxes = self._detect_boxes(frame, imgsz=1024, conf=0.65)
            boxes = self._nms(boxes)  
            
            is_video = source == "webcam" or source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            if not is_video:
                boxes += self._detect_tiled(frame, conf=0.25)
            

            boxes = self._nms(boxes, iou_thresh=0.25)


            raw_dets = []
            for (x1, y1, x2, y2, conf) in boxes:
                bw, bh = x2-x1, y2-y1
                if bh == 0: continue
                ar = bw / bh
                # Nhận diện cả biển vuông và biển nghiêng
                if ar < 0.7 or ar > 7.0: continue 
                if bw < 15 or bh < 8: continue

                x1p=max(0,x1-4); y1p=max(0,y1-4)
                x2p=min(w_f,x2+4); y2p=min(h_f,y2+4)
                crop_bgr = frame[y1p:y2p, x1p:x2p]
                if crop_bgr.size == 0: continue
                pil_crop = Image.fromarray(
                    cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                raw_dets.append((x1p,y1p,x2p,y2p, conf, pil_crop))

            # ── Update tracker: truyền crop vào để tích lũy pool ────
            det_tuples = [(d[0],d[1],d[2],d[3],d[4],
                           "",        # text chưa có, sẽ OCR sau
                           d[5])      
                          for d in raw_dets]
            with self._tracks_lock:
                evicted = self._tracker.yolo_update_with_crops(det_tuples) or []

            
            def _async_ocr_task(trk, best_pil, best_conf, src):
                text = ocr_plate(best_pil, track_id=trk.id)
                
                with self._tracks_lock:
                    trk.text = text
                    trk.best_text = text
                    trk.ocr_state = "done"
                    bbox = tuple(int(v) for v in trk.target)
                
                
                
                try:
                    self._q_result.put_nowait([{
                        "plate_text": text,
                        "confidence": best_conf,
                        "pil_crop":   best_pil,
                        "source":     src,
                        "time":       datetime.now().strftime("%H:%M:%S"),
                        "bbox":       bbox,
                    }])
                except queue.Full: 
                    logging.warning("Hàng đợi Result Queue bị tràn, đã xảy ra hiện tượng rớt gói tin.")

            # ── OCR evicted tracks (xe ra khỏi frame chưa kịp OCR) ───
            for trk in evicted:
                best_pil, best_conf = trk.best_crop()
                if best_pil is None: continue
                # Cấp phát tác vụ vào hàng đợi của ThreadPoolExecutor
                self.ocr_pool.submit(_async_ocr_task, trk, best_pil, best_conf, source)

            # ── OCR deferred: chỉ OCR khi track đủ rõ ───────────
            with self._tracks_lock:
                pending_tracks = [t for t in self._tracker.tracks if t.ready_for_ocr()]
                for trk in pending_tracks:
                    trk.ocr_state = "sent"   # đánh dấu đang xử lý

            for trk in pending_tracks:
                best_pil, best_conf = trk.best_crop()
                if best_pil is None: continue
                # Cấp phát tác vụ vào hàng đợi của ThreadPoolExecutor
                self.ocr_pool.submit(_async_ocr_task, trk, best_pil, best_conf, source)

    def _detect_boxes(self, frame, imgsz=1280, conf=0.50):
        h_f, w_f = frame.shape[:2]
        results = model_yolo(frame, verbose=False, imgsz=imgsz, conf=conf)[0]
        out = []
        for box in results.boxes:
            c = float(box.conf[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            out.append((max(0,x1),max(0,y1),min(w_f,x2),min(h_f,y2),c))
        return out

    def _detect_tiled(self, frame, conf=0.50):
        h_f, w_f = frame.shape[:2]
        cols,rows,overlap = 3,2,0.20
        tw=int(w_f/(cols-overlap*(cols-1)))
        th=int(h_f/(rows-overlap*(rows-1)))
        sx=int(tw*(1-overlap)); sy=int(th*(1-overlap))
        out=[]
        for r in range(rows):
            for c in range(cols):
                tx1,ty1=c*sx,r*sy
                tx2,ty2=min(w_f,tx1+tw),min(h_f,ty1+th)
                tile=frame[ty1:ty2,tx1:tx2]
                if tile.size==0: continue
                for(bx1,by1,bx2,by2,bc) in self._detect_boxes(tile,imgsz=640,conf=conf):
                    out.append((tx1+bx1,ty1+by1,tx1+bx2,ty1+by2,bc))
        return out

    def _detect_smart(self, frame, conf=0.50):
        # Thử detect nhanh trước
        boxes = self._detect_boxes(frame, conf=conf)

        # Chỉ tiled khi không thấy gì
        if not boxes:
            boxes = self._detect_tiled(frame, conf=conf)
            boxes = self._nms(boxes)

        return boxes

    
    @staticmethod
    def _nms(boxes, iou_thresh=0.45):
        if not boxes: return []
        boxes=sorted(boxes,key=lambda b:b[4],reverse=True)
        keep=[]; used=[False]*len(boxes)
        for i,b in enumerate(boxes):
            if used[i]: continue
            keep.append(b)
            for j in range(i+1,len(boxes)):
                if used[j]: continue
                ix1=max(b[0],boxes[j][0]); iy1=max(b[1],boxes[j][1])
                ix2=min(b[2],boxes[j][2]); iy2=min(b[3],boxes[j][3])
                inter=max(0,ix2-ix1)*max(0,iy2-iy1)
                aa=(b[2]-b[0])*(b[3]-b[1]); ab=(boxes[j][2]-boxes[j][0])*(boxes[j][3]-boxes[j][1])
                if inter/max(1,aa+ab-inter)>iou_thresh: used[j]=True
        return keep

    # ═══════════════════════════════════════════════
    #  UI POLL LOOPS
    # ═══════════════════════════════════════════════
    # ═══════════════════════════════════════════════
    #  UI POLL LOOPS
    # ═══════════════════════════════════════════════
    def _poll_display(self):
        # 1. ĐÓNG BĂNG Tracker khi Pause để khung không bị trôi tự do
        if not getattr(self, 'is_paused', False):
            with self._tracks_lock:
                tracks = self._tracker.step(dt=0.030)
                self._active_tracks = tracks
        else:
            tracks = getattr(self, '_active_tracks', [])

        latest = None
        while True:
            try: latest = self._q_display.get_nowait()
            except queue.Empty: break

        # 2. LƯU TRỮ ẢNH GỐC SẠCH (Chưa bị vẽ khung)
        if latest is not None:
            self._clean_frame = latest.copy()

        # 3. Kéo ảnh gốc ra để render
        render = None
        if getattr(self, '_clean_frame', None) is not None:
            render = self._clean_frame.copy()
            if tracks:
                render = draw_tracks(render, tracks)

        if render is not None:
            self._show_frame(render)

        self.after(30, self._poll_display)

    def _poll_results(self):
        while True:
            try:
                cards = self._q_result.get_nowait()
                for det in cards:
                    text = det["plate_text"]

                    if text == "[Không đọc được]" or len(text) < 6 or not re.search(r'\d{2}', text):
                        continue
                    
                    if text not in self.recent_plates:
                        self._add_plate_card(det)
                        # Đã thay current_time thành True để khắc phục lỗi NameError
                        self.recent_plates[text] = True 
            except queue.Empty:
                break
        self.after(200, self._poll_results)

    # ═══════════════════════════════════════════════
    #  DISPLAY
    # ═══════════════════════════════════════════════
    def _show_frame(self, frame_bgr: np.ndarray):
        if frame_bgr is None: return
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 4 or ch < 4: return
        
        h, w   = frame_bgr.shape[:2]
        scale  = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        if nw <= 0 or nh <= 0: return

        frame_resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        
        tk_img = ImageTk.PhotoImage(pil)
        self._photo_refs["main"] = tk_img
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=tk_img)

    def _redraw_last(self):
        # Render lại khi người dùng kéo giãn kích thước cửa sổ App
        if getattr(self, '_clean_frame', None) is not None:
            render = self._clean_frame.copy()
            if getattr(self, '_active_tracks', None):
                render = draw_tracks(render, self._active_tracks)
            self._show_frame(render)

    # ═══════════════════════════════════════════════
    #  DETAIL PANEL
    # ═══════════════════════════════════════════════
    def _select_plate(self, det: dict):
        self.selected_plate = det
        for w in self.detail_scroll.winfo_children():
            w.destroy()

        row = 0
        try:
            pil = det["pil_crop"].copy()
            pil.thumbnail((248, 80))
            tk_img = ctk.CTkImage(light_image=pil, size=pil.size)
            self._photo_refs["detail_crop"] = tk_img
            ctk.CTkLabel(self.detail_scroll, image=tk_img,
                         text="").grid(row=row, column=0, padx=8,
                                       pady=(10, 4), sticky="ew")
            row += 1
        except Exception:
            pass

        ctk.CTkLabel(self.detail_scroll, text=det["plate_text"],
                     font=ctk.CTkFont("Courier New", 22, "bold"),
                     text_color="#00e5ff").grid(row=row, column=0, pady=(4, 8))
        row += 1

        fields = [
            ("Độ tin cậy", f"{det['confidence']:.1%}"),
            ("Thời gian",  det["time"]),
            ("Nguồn",      os.path.basename(det["source"])
                           if det["source"] != "webcam" else "Webcam"),
            ("Tọa độ",     f"({det['bbox'][0]}, {det['bbox'][1]})"),
            ("Kích thước",
             f"{det['bbox'][2]-det['bbox'][0]} × {det['bbox'][3]-det['bbox'][1]} px"),
        ]
        for key, val in fields:
            rf = ctk.CTkFrame(self.detail_scroll, fg_color="#131820",
                              corner_radius=4)
            rf.grid(row=row, column=0, sticky="ew", padx=4, pady=2)
            rf.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(rf, text=key, font=ctk.CTkFont("Courier New", 11),
                         text_color="#4a90a4"
                         ).grid(row=0, column=0, padx=8, pady=5, sticky="w")
            ctk.CTkLabel(rf, text=val, font=ctk.CTkFont("Courier New", 11),
                         text_color="#e8edf5", anchor="e"
                         ).grid(row=0, column=1, padx=8, pady=5, sticky="e")
            row += 1

        self.btn_download.grid(row=2, column=0, padx=10, pady=(6, 10),
                               sticky="ew")

    def _download_crop(self):
        if not self.selected_plate: return
        det  = self.selected_plate
        name = det["plate_text"].replace(" ", "_").replace("/", "-")
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            initialfile=f"plate_{name}_{det['time'].replace(':','')}.jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if path:
            det["pil_crop"].save(path)
            self._set_status(f"✅ Đã lưu: {os.path.basename(path)}")

    # ═══════════════════════════════════════════════
    #  MISC
    # ═══════════════════════════════════════════════
    def _set_status(self, text: str):
        self.after(0, self.lbl_status.configure, {"text": text})

    def on_closing(self):
        self._reader_running = False
        self._yolo_running   = False
        if self.cap: self.cap.release()
        self.destroy()


# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════
if __name__ == "__main__":
    app = LRPApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()