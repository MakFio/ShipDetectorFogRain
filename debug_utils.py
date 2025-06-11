import os
import cv2
import numpy as np
import time
import uuid
from datetime import datetime
from model_fog import FogDetector
from model_coco import CocoDetector
from preprocessing.enhancer import WeatherEnhancer

# Создаем папку для отладочных отчетов
DEBUG_OUTPUT_DIR = "debug_reports"
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)


class DebugProcessor:
    def __init__(self):
        self.fog_detector = FogDetector()
        self.coco_detector = CocoDetector()
        self.weather_enhancer = WeatherEnhancer()
        self.report_id = None
        self.start_time = None
        self.steps = []

    def start_debug_session(self):
        """Инициализировать новую сессию отладки"""
        self.report_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.steps = []
        return self.report_id

    def add_step(self, name, image, data=None):
        """Добавить шаг обработки в отчет"""
        step = {
            "name": name,
            "image": image,
            "timestamp": time.time() - self.start_time,
            "data": data or {}
        }
        self.steps.append(step)
        return step

    def save_image(self, image, filename):
        """Сохранить изображение на диск"""
        cv2.imwrite(filename, image)
        return filename

    def draw_detections(self, image, detections, color=(0, 255, 0)):
        """Нарисовать детекции на изображении"""
        output = image.copy()
        for det in detections:
            x1 = int(det['xmin'] * output.shape[1])
            y1 = int(det['ymin'] * output.shape[0])
            x2 = int(det['xmax'] * output.shape[1])
            y2 = int(det['ymax'] * output.shape[0])
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class']} {det['confidence']:.1f}%"
            cv2.putText(output, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return output

    def process_frame_debug(self, frame):
        """Обработать кадр с детальной отладкой каждого этапа"""
        if self.report_id is None:
            self.start_debug_session()

        # Шаг 1: Оригинальный кадр
        orig_frame = frame.copy()
        self.add_step("Original Frame", orig_frame)

        # Шаг 2: Предобработка MPRNet
        try:
            enhanced_frame = self.weather_enhancer(orig_frame)
            self.add_step("MPRNet Enhanced", enhanced_frame)
        except Exception as e:
            enhanced_frame = orig_frame
            self.add_step("MPRNet Failed", orig_frame, {"error": str(e)})

        # Шаг 3: Детекция FogDetector
        fog_detections = self.fog_detector.process_frame(enhanced_frame)
        for det in fog_detections:
            det['class'] = 'ship'
        fog_vis = self.draw_detections(enhanced_frame.copy(), fog_detections, (0, 255, 0))
        self.add_step("FogDetector Results", fog_vis, {
            "detections": fog_detections,
            "count": len(fog_detections),
            "avg_confidence": np.mean([d['confidence'] for d in fog_detections]) if fog_detections else 0
        })

        # Шаг 4: Детекция CocoDetector
        coco_detections = self.coco_detector.process_frame(enhanced_frame)
        for det in coco_detections:
            det['class'] = 'boat'
        coco_vis = self.draw_detections(enhanced_frame.copy(), coco_detections, (0, 0, 255))
        self.add_step("CocoDetector Results", coco_vis, {
            "detections": coco_detections,
            "count": len(coco_detections),
            "avg_confidence": np.mean([d['confidence'] for d in coco_detections]) if coco_detections else 0
        })

        # Шаг 5: Сравнение моделей
        fog_conf = np.mean([d['confidence'] for d in fog_detections]) if fog_detections else 0
        coco_conf = np.mean([d['confidence'] for d in coco_detections]) if coco_detections else 0
        selected_model = 'fog' if fog_conf > coco_conf else 'coco'
        final_detections = fog_detections if selected_model == 'fog' else coco_detections

        comp_vis = enhanced_frame.copy()
        comp_vis = self.draw_detections(comp_vis, fog_detections, (0, 255, 0))  # Зеленый для Fog
        comp_vis = self.draw_detections(comp_vis, coco_detections, (0, 0, 255))  # Красный для COCO
        self.add_step("Model Comparison", comp_vis, {
            "fog_confidence": fog_conf,
            "coco_confidence": coco_conf,
            "selected_model": selected_model
        })

        # Шаг 6: Финальный результат
        final_vis = self.draw_detections(enhanced_frame.copy(), final_detections,
                                         (0, 255, 0) if selected_model == 'fog' else (0, 0, 255))
        self.add_step("Final Detection", final_vis, {
            "detections": final_detections,
            "count": len(final_detections)
        })

        # Генерируем отчет
        report_path = self.generate_report()

        return final_vis, final_detections, report_path

    def generate_report(self):
        """Сгенерировать HTML отчет с результатами отладки"""
        if not self.steps:
            return None

        report_dir = os.path.join(DEBUG_OUTPUT_DIR, self.report_id)
        os.makedirs(report_dir, exist_ok=True)

        # Сохраняем изображения
        step_images = []
        for i, step in enumerate(self.steps):
            img_name = f"step_{i}_{step['name'].replace(' ', '_')}.jpg"
            img_path = os.path.join(report_dir, img_name)
            self.save_image(step['image'], img_path)
            # Сохраняем только имя файла без пути
            step_images.append((step, img_name))

        # Создаем HTML отчет
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Debug Report - {self.report_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .report-header {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .step {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .step-header {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
                .step-image {{ max-width: 100%; max-height: 600px; border: 1px solid #ccc; }}
                .step-data {{ background-color: #f9f9f9; padding: 10px; margin-top: 10px; border-radius: 3px; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <h1>Debug Report</h1>
                <p>ID: {self.report_id}</p>
                <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """

        for i, (step, img_filename) in enumerate(step_images):
            html_report += f"""
            <div class="step">
                <div class="step-header">
                    Step {i + 1}: {step['name']}
                    <span class="timestamp">({step['timestamp']:.2f}s)</span>
                </div>
                <img src="{img_filename}" alt="{step['name']}" class="step-image">
                <div class="step-data">
                    <pre>{self.format_step_data(step['data'])}</pre>
                </div>
            </div>
            """

        html_report += "</body></html>"

        # Сохраняем HTML отчет
        report_file = os.path.join(report_dir, "report.html")
        with open(report_file, "w", encoding='utf-8') as f:
            f.write(html_report)

        return report_file

    def format_step_data(self, data):
        """Форматировать данные шага для отображения"""
        if not data:
            return "No additional data"

        if isinstance(data, dict):
            return "\n".join(f"{k}: {v}" for k, v in data.items())

        return str(data)


# Глобальный экземпляр для отладки
debug_processor = DebugProcessor()