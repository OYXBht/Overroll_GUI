# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
from functools import partial
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
import threading
import os

class EnhancedTrafficSystem:
    def __init__(self, master):
        self.master = master
        master.title("恶劣天气交通标识增强检测系统")
        master.geometry("1200x800")

        # 初始化参数系统
        self.enhance_params = {
            'enable_wb': tk.BooleanVar(value=True),
            'wb_strength': tk.DoubleVar(value=0.87),
            'enable_clahe': tk.BooleanVar(value=False),
            'clahe_clip': tk.DoubleVar(value=1.0),
            'enable_haze': tk.BooleanVar(value=True),
            'haze_threshold': tk.DoubleVar(value=0.15),
            'enable_sharp': tk.BooleanVar(value=True),
            'sharp_strength': tk.DoubleVar(value=1.12),
            'atmospheric_percent': tk.DoubleVar(value=60)
        }

        self.detect_params = {
            'conf_thres': tk.DoubleVar(value=0.3),
            'iou_thres': tk.DoubleVar(value=0.4)
        }

        # 初始化模型和界面
        self.setup_model()
        self.create_interface()
        self.setup_bindings()

    def setup_model(self):
        set_logging()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 确保模型路径正确
        model_path = os.path.join(os.getcwd(), 'D:\\PycharmProjects\\Overroll_GUI\\other_models\\best.pt')
        if not os.path.exists(model_path):
            print("错误：模型文件不存在，请检查路径")
            return
        self.model = attempt_load(model_path, map_location=self.device)
        self.model.half().to(self.device).eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def create_interface(self):
        # 主布局框架
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 控制面板
        control_frame = ttk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # 图像增强控制
        enhance_frame = ttk.LabelFrame(control_frame, text="图像增强参数")
        enhance_frame.pack(fill=tk.X, pady=5)
        self.create_enhance_controls(enhance_frame)

        # 目标检测控制
        detect_frame = ttk.LabelFrame(control_frame, text="目标检测参数")
        detect_frame.pack(fill=tk.X, pady=5)
        self.create_detect_controls(detect_frame)

        # 功能按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="打开图片", command=self.load_image).pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="图像增强", command=self.enhance_image_only).pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="目标检测", command=self.detect_image_only).pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="增强并检测", command=self.enhance_and_detect).pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="保存结果", command=self.save_image).pack(side=tk.TOP, fill=tk.X, pady=5)

        # 图像显示区
        img_display_frame = ttk.Frame(main_frame)
        img_display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # 第一行：原图和增强图
        top_row_frame = ttk.Frame(img_display_frame)
        top_row_frame.pack(fill=tk.BOTH, expand=True)

        self.original_canvas = self.create_canvas(top_row_frame, "原始图像")
        self.enhanced_canvas = self.create_canvas(top_row_frame, "增强结果")

        # 第二行：检测图
        bottom_row_frame = ttk.Frame(img_display_frame)
        bottom_row_frame.pack(fill=tk.BOTH, expand=True)

        self.detected_canvas = self.create_canvas(bottom_row_frame, "检测结果")

        # 统计信息
        self.stats_label = ttk.Label(self.master, font=('微软雅黑', 10))
        self.stats_label.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_bindings(self):
        self.master.bind('<Return>', lambda e: self.enhance_and_detect())

    def create_enhance_controls(self, parent):
        self.create_switch(parent, "白平衡", 'enable_wb')
        self.create_param_control(parent, "白平衡强度", 'wb_strength', 0.0, 2.0)
        self.create_switch(parent, "CLAHE增强", 'enable_clahe')
        self.create_param_control(parent, "CLAHE对比度", 'clahe_clip', 0.0, 5.0)
        self.create_switch(parent, "去雾处理", 'enable_haze')
        self.create_param_control(parent, "去雾强度", 'haze_threshold', 0.0, 1.0)
        self.create_param_control(parent, "大气光百分比", 'atmospheric_percent', 0.0, 99.9)
        self.create_switch(parent, "锐化处理", 'enable_sharp')
        self.create_param_control(parent, "锐化强度", 'sharp_strength', 0.0, 3.0)

    def create_detect_controls(self, parent):
        self.create_param_control(parent, "置信度阈值", 'conf_thres', 0.0, 1.0)
        self.create_param_control(parent, "IoU阈值", 'iou_thres', 0.0, 1.0)

    def get_var(self, param):
        """统一获取参数变量"""
        if param in self.enhance_params:
            return self.enhance_params[param]
        return self.detect_params[param]

    def create_param_control(self, parent, label, param, min_val, max_val):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT)

        var = self.get_var(param)
        entry = ttk.Entry(frame, width=6, textvariable=var)
        entry.pack(side=tk.LEFT, padx=5)

        slider = ttk.Scale(frame, variable=var, from_=min_val, to=max_val, orient=tk.HORIZONTAL)
        slider.pack(side=tk.LEFT, expand=True)

    def create_switch(self, parent, label, param):
        var = self.get_var(param)
        ttk.Checkbutton(parent, text=label, variable=var).pack(anchor=tk.W)

    def create_canvas(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        canvas = tk.Canvas(frame, width=400, height=300, bg='#F0F0F0')
        canvas.pack(fill=tk.BOTH, expand=True)
        return canvas

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            self.original_img = cv2.imread(path)
            if self.original_img is None:
                print("错误：无法加载图像，请检查文件路径和格式。")
                return
            self.show_image(self.original_img, self.original_canvas)
            # 清空其他显示
            self.enhanced_canvas.delete("all")
            self.detected_canvas.delete("all")
            self.stats_label.config(text="")

    def enhance_image_only(self):
        if not hasattr(self, 'original_img'):
            return
        self.toggle_ui_state(False)
        threading.Thread(target=self._enhance_image_only, daemon=True).start()

    def _enhance_image_only(self):
        try:
            enhanced = self.enhance_image(self.original_img.copy())
            self.enhanced_img = enhanced
            self.master.after(0, lambda: self.show_image(enhanced, self.enhanced_canvas))
            self.master.after(0, lambda: self.stats_label.config(text="图像增强完成"))
        # except Exception as e:
        #     self.master.after(0, lambda: self.stats_label.config(text=f"增强错误: {str(e)}"))
        except Exception as e:
            print("[增强错误]" + str(e))
            self.master.after(0, lambda e=e: self.stats_label.config(text=f"增强错误: {str(e)}"))  # 显式传递 e
        finally:
            self.master.after(0, lambda: self.toggle_ui_state(True))

    def detect_image_only(self):
        if not hasattr(self, 'original_img'):
            return
        self.toggle_ui_state(False)
        threading.Thread(target=self._detect_image_only, daemon=True).start()

    def _detect_image_only(self):
        try:
            detected, detections = self.detect_objects(self.original_img.copy())
            self.master.after(0, lambda: self.show_image(detected, self.detected_canvas))
            self.update_stats(detections)
        # except Exception as e:
        #     self.master.after(0, lambda: self.stats_label.config(text=f"检测错误: {str(e)}"))
        except Exception as e:
            print("[检测错误]" + str(e))
            self.master.after(0, lambda e=e: self.stats_label.config(text=f"检测错误: {str(e)}"))  # 显式传递 e
        finally:
            self.master.after(0, lambda: self.toggle_ui_state(True))

    def enhance_and_detect(self):
        if not hasattr(self, 'original_img'):
            return
        self.toggle_ui_state(False)
        threading.Thread(target=self._enhance_and_detect, daemon=True).start()

    def _enhance_and_detect(self):
        try:
            # 图像增强
            enhanced = self.enhance_image(self.original_img.copy())
            self.enhanced_img = enhanced
            # 目标检测
            detected, detections = self.detect_objects(enhanced)

            self.master.after(0, lambda: self.show_image(enhanced, self.enhanced_canvas))
            self.master.after(0, lambda: self.show_image(detected, self.detected_canvas))
            self.update_stats(detections)
        # except Exception as e:
        #     self.master.after(0, lambda: self.stats_label.config(text=f"处理错误: {str(e)}"))
        except Exception as e:
            print("[处理错误]" + str(e))
            self.master.after(0, lambda e=e: self.stats_label.config(text=f"处理错误: {str(e)}"))  # 显式传递 e
        finally:
            self.master.after(0, lambda: self.toggle_ui_state(True))

    def enhance_image(self, img):
        img = img.astype(np.float32)
        # 白平衡处理
        if self.enhance_params['enable_wb'].get():
            img = self.white_balance(img)
        # CLAHE增强
        if self.enhance_params['enable_clahe'].get():
            img = self.apply_clahe(img)
        # 去雾处理
        if self.enhance_params['enable_haze'].get():
            img = self.dehaze(img)
        # 锐化处理
        if self.enhance_params['enable_sharp'].get():
            img = self.sharpen(img)
        return np.clip(img, 0, 255).astype(np.uint8)

    def white_balance(self, img):
        avg_b = np.mean(img[:, :, 0])
        avg_g = np.mean(img[:, :, 1])
        avg_r = np.mean(img[:, :, 2])
        avg = (avg_b + avg_g + avg_r) / 3
        scale = np.array([avg / avg_b, avg / avg_g, avg / avg_r]) * self.enhance_params['wb_strength'].get()
        img = img * scale[np.newaxis, np.newaxis, :]
        return np.clip(img, 0, 255)

    def apply_clahe(self, img):
        lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=self.enhance_params['clahe_clip'].get(),
            tileGridSize=(8, 8)
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    def dehaze(self, img):
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        dark = cv2.erode(gray, np.ones((15, 15), np.uint8))
        atmospheric_light = np.percentile(dark, 100 - self.enhance_params['atmospheric_percent'].get())
        transmission = 1 - self.enhance_params['haze_threshold'].get() * (gray.astype(np.float32) / atmospheric_light)
        transmission = np.clip(transmission, 0.1, 0.9)
        # transmission = cv2.ximgproc.guidedFilter(gray, transmission, 15, 1e-3)
        transmission = cv2.blur(transmission,( 15, 15))
        result = np.empty_like(img)
        for i in range(3):
            result[:, :, i] = (img[:, :, i] - atmospheric_light) / transmission + atmospheric_light
        return np.clip(result, 0, 255)

    def sharpen(self, img):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel * self.enhance_params['sharp_strength'].get())

    def detect_objects(self, img):
        img_preprocessed, ratio, pad = self.letterbox(img, new_shape=640)
        img_tensor = torch.from_numpy(img_preprocessed).to(self.device)
        img_tensor = img_tensor.half().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = self.model(img_tensor)[0]

        pred = non_max_suppression(pred,
                                   conf_thres=self.detect_params['conf_thres'].get(),
                                   iou_thres=self.detect_params['iou_thres'].get())

        detections = []
        if pred[0] is not None:
            det = pred[0]
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{self.names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img, label=label, color=(32, 165, 218), line_thickness=2)
                detections.append((self.names[int(cls)], float(conf)))
        return img, detections

    def show_image(self, img, canvas):
        h, w = img.shape[:2]
        ratio = min(400 / w, 400 / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(img_pil)
        canvas.image = img_tk
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2, anchor=tk.CENTER, image=img_tk)

    def update_stats(self, detections):
        if not detections:
            text = "检测结果：未发现目标"
        else:
            total = len(detections)
            confidences = [d[1] for d in detections]
            avg_conf = sum(confidences) / total
            classes = ', '.join(set([d[0] for d in detections]))
            text = f"检测到 {total} 个目标 | 平均置信度：{avg_conf:.2f} | 类别：{classes}"
        self.stats_label.config(text=text)

    def toggle_ui_state(self, enabled):
        state = 'normal' if enabled else 'disabled'
        for child in self.master.winfo_children():
            if isinstance(child, (ttk.Button, ttk.Entry)):
                child['state'] = state

    def save_image(self):
        if hasattr(self, 'enhanced_img'):
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                cv2.imwrite(path, self.enhanced_img)
                self.stats_label.config(text=f"图像已保存到: {path}")

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleup:
            dw, dh = 0.0, 0.0

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(im, int(dh), int(dh), int(dw), int(dw),
                                cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.configure('TButton', font=('微软雅黑', 10))
    style.configure('TLabelFrame', font=('微软雅黑', 10))
    app = EnhancedTrafficSystem(root)
    root.mainloop()
