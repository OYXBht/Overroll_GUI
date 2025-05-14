import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import os
import torch
from deeplabv3.deeplab import DeeplabV3
from Yolov7.yolo import YOLO
from Lane_line.lane import Lane_line
from utils.general import set_logging
from imutils.object_detection import non_max_suppression
import imutils

try:
    from buzzer import check_distance_and_buzz
    BUZZER_AVAILABLE = True
except ImportError:
    BUZZER_AVAILABLE = False

print(BUZZER_AVAILABLE)

# ==== 模型配置 (在此修改权重路径) ====
YOLO_WEIGHTS = "./models/yolov7_default.pt"
DEEP_MODEL_PATH = "./models/deeplabv3_default.pth"


class UnifiedTrafficApp:
    def __init__(self, master):
        self.master = master
        master.title("交通场景统一检测系统")
        master.geometry("1300x800")

        # 初始化模型
        self._init_models()
        # GUI 布局
        self._create_interface()

    def _init_models(self):
        self.yolo = YOLO()
        try:
            self.yolo.model_path = YOLO_WEIGHTS
            self.yolo.load_model()
        except Exception:
            pass

        self.deeplab = DeeplabV3()
        self.deeplab._defaults["model_path"] = DEEP_MODEL_PATH
        try:
            self.deeplab.load_model()
        except Exception:
            pass
        self.lane = Lane_line()

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        set_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_interface(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control = ttk.Frame(main_frame, width=350)
        control.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Label(control, text="文件选择:").pack(anchor=tk.W, pady=(0, 2))
        ttk.Button(control, text="打开图像/视频", command=self.load_path).pack(fill=tk.X, pady=5)
        self.path_var = tk.StringVar()
        ttk.Entry(control, textvariable=self.path_var, state='readonly').pack(fill=tk.X)

        # 图像识别功能区域
        ttk.Label(control, text="图像识别:").pack(anchor=tk.W, pady=(10, 2))
        self.mode_lane = tk.BooleanVar(value=False)
        self.mode_sign = tk.BooleanVar(value=False)
        # self.mode_people = tk.BooleanVar(value=False)
        self.mode_traffic = tk.BooleanVar(value=False)
        self.mode_zebra = tk.BooleanVar(value=False)
        ttk.Checkbutton(control, text="车道线检测", variable=self.mode_lane).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(control, text="交通场景元素识别", variable=self.mode_sign).pack(anchor=tk.W, pady=2)
        # ttk.Checkbutton(control, text="行人检测", variable=self.mode_people).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(control, text="红绿灯识别", variable=self.mode_traffic).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(control, text="斑马线识别", variable=self.mode_zebra).pack(anchor=tk.W, pady=2)

        btn_frame = ttk.Frame(control)
        btn_frame.pack(fill=tk.X, pady=15)
        ttk.Button(btn_frame, text="开始识别", command=self.run_pipeline).pack(side=tk.LEFT, expand=True)
        ttk.Button(btn_frame, text="保存结果", command=self.save_result).pack(side=tk.LEFT, expand=True)

        # 设别开关区域
        ttk.Label(control, text="设备控制:").pack(anchor=tk.W, pady=(15,2))
        self.mode_buzzer = tk.BooleanVar(value=False)
        ttk.Checkbutton(control, text='蜂鸣器开关' if BUZZER_AVAILABLE else '蜂鸣器开关/设备未发现',
                        variable=self.mode_buzzer,
                        state='normal' if BUZZER_AVAILABLE else 'disabled').pack(anchor=tk.W, pady=2)

        display = ttk.Frame(main_frame)
        display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(display, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status = ttk.Label(self.master, text="等待操作...", anchor=tk.W)
        self.status.pack(fill=tk.X)

    def load_path(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image/Video", "*.jpg *.jpeg *.png *.bmp *.mp4 *.mov")]
        )
        if path:
            self.path_var.set(path)
            self.status.config(text="已选择: " + os.path.basename(path))

            # 显示初始图像
            if any(path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (1000, 600))
                    self._show(img)
                else:
                    self.status.config(text="无法加载图像")

    def run_pipeline(self):
        if not self.path_var.get(): return
        threading.Thread(target=self._process, daemon=True).start()

    def _process(self):
        path = self.path_var.get()
        if any(path.lower().endswith(ext) for ext in ['.mp4','.mov']):
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            if not ret:
                self.status.config(text="读取视频失败")
                return
            img = frame
            cap.release()
        else:
            img = cv2.imread(path)
            if img is None:
                self.status.config(text="读取图像失败")
                return

        img = cv2.resize(img, (1000, 600))
        base_img = img.copy()
        overlay = np.zeros_like(base_img)

        traffic_color = ""
        if self.mode_traffic.get():
            hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
            red_mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
            red_mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
            green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
            red_detected = cv2.countNonZero(red_mask)
            yellow_detected = cv2.countNonZero(yellow_mask)
            green_detected = cv2.countNonZero(green_mask)
            if red_detected > yellow_detected and red_detected > green_detected:
                traffic_color = "Red"
            elif yellow_detected > red_detected and yellow_detected > green_detected:
                traffic_color = "Yellow"
            elif green_detected > red_detected and green_detected > yellow_detected:
                traffic_color = "Green"
            else:
                traffic_color = "Unknown"
            cv2.putText(overlay, f"Traffic Light: {traffic_color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 行人检测（对原图尺寸执行检测，避免因缩放错位）
        # if self.mode_people.get():
        #     (rects, _) = self.hog.detectMultiScale(base_img, winStride=(4, 4), padding=(8, 8), scale=1.05)
        #     rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        #     pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        #     for (xA, yA, xB, yB) in pick:
        #         cv2.rectangle(overlay, (xA, yA), (xB, yB), (0, 255, 0), 2)

        if self.mode_lane.get():
            pil = Image.fromarray(base_img)
            _, binary = self.deeplab.detect_image(pil)
            base_img = self._draw_lane(base_img, binary)

        if self.mode_sign.get():
            pil2 = Image.fromarray(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
            base_img = np.array(self.yolo.detect_image(pil2))

        result = cv2.addWeighted(base_img, 1.0, overlay, 1.0, 0)

        if self.mode_zebra.get():
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 10)
            _, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
            eroded = cv2.erode(thresh, np.ones((3, 1), np.uint8), iterations=3)
            dilated = cv2.dilate(eroded, np.ones((5, 1), np.uint8), iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if x > 100 and w > 50 and h > 10:
                    cv2.drawContours(result, [c], -1, (255, 0, 0), 2)
            cv2.putText(result, "Zebra Crossing Detected", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        self._show(result)
        self.status.config(text="识别完成")

        if self.mode_buzzer.get() and BUZZER_AVAILABLE:
            try:
                check_distance_and_buzz(threshold=10.0)
            except Exception:
                pass

    def _detect_people(self, img):
        resized = imutils.resize(img, width=min(600, img.shape[1]))
        (rects, _) = self.hog.detectMultiScale(resized, winStride=(4, 4), padding=(8, 8), scale=1.05)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(resized, (xA, yA), (xB, yB), (0, 255, 0), 2)
        return resized

    def _detect_traffic_light(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))

        red_detected = cv2.countNonZero(red_mask)
        yellow_detected = cv2.countNonZero(yellow_mask)
        green_detected = cv2.countNonZero(green_mask)

        if red_detected > yellow_detected and red_detected > green_detected:
            color = "Red"
        elif yellow_detected > red_detected and yellow_detected > green_detected:
            color = "Yellow"
        elif green_detected > red_detected and green_detected > yellow_detected:
            color = "Green"
        else:
            color = "Unknown"

        cv2.putText(img, f"Traffic Light: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    def _draw_lane(self, img, binary):
        src = [(415, 335), (585, 335), (1000, 600), (0, 600)]
        dst = [(0, 0), (1000, 0), (1000, 600), (0, 600)]
        M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
        N = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
        warped = self.lane.perspective_transform(binary, M)
        lf, rf, _, tf = self.lane.find_line_fit(np.array(warped))
        if tf != 1:
            return img
        lx, rx, py = self.lane.get_fit_xy(warped, lf, rf)
        back = self.lane.project_back(np.array(warped), lx, rx, py)
        newwarp = self.lane.perspective_transform(back, N)
        return cv2.addWeighted(img, 1, newwarp, 0.7, 0)

    def _show(self, img):
        h, w = img.shape[:2]
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        ratio = min(cw / w, ch / h) if cw and ch else 1
        resized = cv2.resize(img, (int(w * ratio), int(h * ratio)))
        img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        self._tkimg = ImageTk.PhotoImage(img_pil)
        self.canvas.delete('all')
        self.canvas.create_image(cw / 2, ch / 2, image=self._tkimg, anchor=tk.CENTER)

    def save_result(self):
        if hasattr(self, '_tkimg'):
            path = filedialog.asksaveasfilename(defaultextension='.png')
            if path:
                # 使用 PIL Image 保存 _tkimg 的图像数据
                img = self._tkimg._PhotoImage__photo.zoom(1).subsample(1)  # 获取底层 Tk PhotoImage
                width = self._tkimg.width()
                height = self._tkimg.height()
                image = Image.frombytes('RGB', (width, height), img.data, 'raw', 'RGB')
                image.save(path)
                self.status.config(text="已保存: " + os.path.basename(path))


if __name__ == '__main__':
    root = tk.Tk()
    app = UnifiedTrafficApp(root)
    root.mainloop()
