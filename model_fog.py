from ultralytics import YOLO
import cv2
import imageio


class FogDetector:
    def __init__(self, model_path='yolov11n_fog.pt'):
        self.model = YOLO(model_path)  # Загрузка дообученной модели
        self.ship_class_id = 0  # В дообученной модели класс "ship" имеет ID 0
        self.ship_class_name = "ship"  # Имя класса

    def process_image(self, input_path, output_path):
        """Обработка статических изображений"""
        results = self.model(input_path)
        results[0].save(output_path)
        return len([box for box in results[0].boxes if box.cls == self.ship_class_id])

    def process_video(self, input_path, output_path):
        """Обработка видеофайлов"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(3))
        height = int(cap.get(4))

        output_path = output_path.rsplit('.', 1)[0] + '.mp4'
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        ship_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(
                source=frame,
                classes=[self.ship_class_id],  # Фильтрация по классу "ship"
                conf=0.5,
                verbose=False
            )

            annotated_frame = results[0].plot()
            writer.write(annotated_frame)

            current = len(results[0].boxes)
            ship_count = max(ship_count, current)

        cap.release()
        writer.release()
        return ship_count

    def process_gif(self, input_path, output_path):
        """Обработка GIF с сохранением метаданных"""
        gif = imageio.get_reader(input_path)
        meta = gif.get_meta_data()

        processed_frames = []
        ship_count = 0

        for frame in gif:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) if frame.shape[2] == 4 else frame
            results = self.model(frame_rgb)
            processed_frames.append(results[0].plot())

            current = len([box for box in results[0].boxes if box.cls == self.ship_class_id])
            ship_count = max(ship_count, current)

        imageio.mimsave(
            output_path,
            processed_frames,
            duration=meta.get('duration', 100),
            loop=meta.get('loop', 0)
        )
        return ship_count

    def process_media(self, input_path, output_path):
        """Универсальный обработчик для всех типов медиа"""
        ext = input_path.split('.')[-1].lower()

        if ext == 'gif':
            return self.process_gif(input_path, output_path)
        elif ext in ['mp4', 'avi', 'mov']:
            return self.process_video(input_path, output_path)
        else:
            raise ValueError("Unsupported media format")

    def process_frame(self, frame):
        """Обработка кадра в реальном времени"""
        results = self.model(frame)
        return [
            {
                'xmin': box.xyxyn[0][0].item(),
                'ymin': box.xyxyn[0][1].item(),
                'xmax': box.xyxyn[0][2].item(),
                'ymax': box.xyxyn[0][3].item(),
                'confidence': box.conf.item() * 100
            }
            for box in results[0].boxes if box.cls == self.ship_class_id  # Фильтрация по классу "ship"
        ]