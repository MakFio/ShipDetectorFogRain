import os
import traceback
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from model_fog import FogDetector
from model_coco import CocoDetector
from preprocessing.enhancer import WeatherEnhancer
import mimetypes
import json
import concurrent.futures
from datetime import datetime
import imageio
from typing import Tuple, List, Dict
from debug_utils import debug_processor

DEBUG_MODE = True

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'EXPORT_FOLDER': 'exports',
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///detections.db',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024
})

db = SQLAlchemy(app)
fog_detector = FogDetector()
coco_detector = CocoDetector()
weather_enhancer = WeatherEnhancer()
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=db.func.now())
    filename = db.Column(db.String(128))
    file_type = db.Column(db.String(10))
    fog_ship_count = db.Column(db.Integer)
    coco_ship_count = db.Column(db.Integer)
    final_ship_count = db.Column(db.Integer)
    fog_detection_data = db.Column(db.Text)
    coco_detection_data = db.Column(db.Text)


def apply_nms(detections: List[Dict], threshold: float = 0.5) -> List[Dict]:
    """
    Применяет Non-Maximum Suppression для подавления вложенных боксов.
    Удаляет детекции, которые сильно перекрываются или полностью содержатся внутри других боксов.
    """
    if not detections:
        return []

    # Конвертируем детекции в формат [x1, y1, x2, y2, confidence]
    boxes = []
    confidences = []
    for det in detections:
        x1 = det['xmin']
        y1 = det['ymin']
        x2 = det['xmax']
        y2 = det['ymax']
        boxes.append([x1, y1, x2, y2])
        confidences.append(det['confidence'])

    boxes = np.array(boxes)
    confidences = np.array(confidences)

    # Применяем NMS с использованием OpenCV
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=confidences.tolist(),
        score_threshold=0.01,
        nms_threshold=threshold
    )

    # Собираем отфильтрованные детекции
    if isinstance(indices, tuple):
        indices = indices[0]
    elif len(indices) == 0:
        return []

    filtered_detections = []
    for i in indices.flatten():
        filtered_detections.append(detections[i])

    return filtered_detections


def is_inside(inner_box: Dict, outer_box: Dict, threshold: float = 0.95) -> bool:
    """
    Проверяет, содержится ли один бокс внутри другого.
    Возвращает True, если более 95% площади внутреннего бокса содержится во внешнем.
    """

    inner_area = (inner_box['xmax'] - inner_box['xmin']) * (inner_box['ymax'] - inner_box['ymin'])

    # Рассчитываем площадь пересечения
    x_left = max(inner_box['xmin'], outer_box['xmin'])
    y_top = max(inner_box['ymin'], outer_box['ymin'])
    x_right = min(inner_box['xmax'], outer_box['xmax'])
    y_bottom = min(inner_box['ymax'], outer_box['ymax'])

    if x_right < x_left or y_bottom < y_top:
        return False

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    return intersection_area / inner_area > threshold


def remove_nested_detections(detections: List[Dict]) -> List[Dict]:
    """
    Удаляет детекции, которые полностью содержатся внутри других боксов.
    """
    if not detections:
        return []

    detections_sorted = sorted(detections,
                               key=lambda d: (d['xmax'] - d['xmin']) * (d['ymax'] - d['ymin']),
                               reverse=True)

    filtered_detections = []
    for i, det in enumerate(detections_sorted):
        is_nested = False

        # Проверяем, содержится ли текущий бокс в любом из уже отфильтрованных
        for kept_det in filtered_detections:
            if is_inside(det, kept_det):
                is_nested = True
                break

        if not is_nested:
            filtered_detections.append(det)

    return filtered_detections


def process_frame_image(frame: np.ndarray) -> Tuple[np.ndarray, int, int, int, List[dict], List[dict], str, List[dict]]:
    original_frame = frame.copy()

    try:
        future = executor.submit(weather_enhancer, frame)
        processed_frame = future.result(timeout=0.9)
    except Exception as e:
        processed_frame = frame

    # Режим отладки
    if DEBUG_MODE:
        debug_processor.start_debug_session()
        output_frame, final_detections, report_path = debug_processor.process_frame_debug(frame)

        return (
            output_frame,
            len(final_detections),
            len(final_detections),  # fog_count
            len(final_detections),  # coco_count
            final_detections,  # fog_detections
            final_detections,  # coco_detections
            "debug",
            final_detections
        )

    fog_future = executor.submit(fog_detector.process_frame, processed_frame)
    coco_future = executor.submit(coco_detector.process_frame, processed_frame)

    fog_detections = fog_future.result()
    coco_detections = coco_future.result()

    fog_detections = remove_nested_detections(fog_detections)
    coco_detections = remove_nested_detections(coco_detections)

    fog_detections = apply_nms(fog_detections)
    coco_detections = apply_nms(coco_detections)

    fog_conf = np.mean([d['confidence'] for d in fog_detections]) if fog_detections else 0
    coco_conf = np.mean([d['confidence'] for d in coco_detections]) if coco_detections else 0

    selected_detections = fog_detections if fog_conf > coco_conf else coco_detections
    selected_model = 'fog' if fog_conf > coco_conf else 'coco'

    # Фильтрация выбранных детекций
    selected_detections = remove_nested_detections(selected_detections)
    selected_detections = apply_nms(selected_detections)

    output_frame = original_frame.copy()
    for det in selected_detections:
        x1 = int(det['xmin'] * output_frame.shape[1])
        y1 = int(det['ymin'] * output_frame.shape[0])
        x2 = int(det['xmax'] * output_frame.shape[1])
        y2 = int(det['ymax'] * output_frame.shape[0])
        color = (0, 255, 0) if selected_model == 'fog' else (0, 0, 255)
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(output_frame,
                    f"{det['confidence']:.1f}% ({selected_model})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1)

    return (output_frame,
            len(selected_detections),
            len(fog_detections),
            len(coco_detections),
            fog_detections,
            coco_detections,
            selected_model,
            selected_detections)


def process_with_both_models(input_path: str, output_path: str, file_type: str) -> Tuple[int, int, int, list, list]:
    if file_type in ['jpg', 'jpeg', 'png']:
        frame = cv2.imread(input_path)
        output_frame, final_count, fog_count, coco_count, fog_dets, coco_dets, _, _ = process_frame_image(frame)
        cv2.imwrite(output_path, output_frame)
        return final_count, fog_count, coco_count, fog_dets, coco_dets

    elif file_type == 'gif':
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data().get('fps', 30)
        skip_frames = max(1, int(fps / 5))
        processed_frames = []

        max_final = 0
        max_fog = 0
        max_coco = 0
        all_fog_dets = []
        all_coco_dets = []

        last_selected_detections = None
        last_selected_model = None
        frame_count = 0

        try:
            for frame in reader:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                output_frame = bgr_frame.copy()

                if frame_count % skip_frames == 0:
                    processed_frame, frame_final, frame_fog, frame_coco, fog_dets, coco_dets, selected_model, selected_detections = process_frame_image(
                        bgr_frame)
                    last_selected_detections = selected_detections
                    last_selected_model = selected_model
                    output_frame = processed_frame

                    max_final = max(max_final, frame_final)
                    max_fog = max(max_fog, frame_fog)
                    max_coco = max(max_coco, frame_coco)
                    all_fog_dets.extend(fog_dets)
                    all_coco_dets.extend(coco_dets)
                else:
                    if last_selected_detections:
                        for det in last_selected_detections:
                            x1 = int(det['xmin'] * output_frame.shape[1])
                            y1 = int(det['ymin'] * output_frame.shape[0])
                            x2 = int(det['xmax'] * output_frame.shape[1])
                            y2 = int(det['ymax'] * output_frame.shape[0])
                            color = (0, 255, 0) if last_selected_model == 'fog' else (0, 0, 255)
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(output_frame,
                                        f"{det['confidence']:.1f}% ({last_selected_model})",
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        color,
                                        1)

                rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                processed_frames.append(rgb_frame)
                frame_count += 1

        except RuntimeError:
            pass
        finally:
            reader.close()

        imageio.mimsave(
            output_path,
            processed_frames,
            format='GIF',
            fps=fps,
            loop=0
        )

        return max_final, max_fog, max_coco, all_fog_dets, all_coco_dets

    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_source = cap.get(cv2.CAP_PROP_FPS)
        skip_frames = max(1, int(fps_source / 5))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_source, (width, height))

        max_final = 0
        max_fog = 0
        max_coco = 0
        all_fog_dets = []
        all_coco_dets = []

        last_selected_detections = None
        last_selected_model = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                output_frame, frame_final, frame_fog, frame_coco, fog_dets, coco_dets, selected_model, selected_detections = process_frame_image(
                    frame)
                last_selected_detections = selected_detections
                last_selected_model = selected_model

                max_final = max(max_final, frame_final)
                max_fog = max(max_fog, frame_fog)
                max_coco = max(max_coco, frame_coco)
                all_fog_dets.extend(fog_dets)
                all_coco_dets.extend(coco_dets)
            else:
                output_frame = frame.copy()
                if last_selected_detections:
                    for det in last_selected_detections:
                        x1 = int(det['xmin'] * output_frame.shape[1])
                        y1 = int(det['ymin'] * output_frame.shape[0])
                        x2 = int(det['xmax'] * output_frame.shape[1])
                        y2 = int(det['ymax'] * output_frame.shape[0])
                        color = (0, 255, 0) if last_selected_model == 'fog' else (0, 0, 255)
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(output_frame,
                                    f"{det['confidence']:.1f}% ({last_selected_model})",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    1)

            out.write(output_frame)
            frame_count += 1

        cap.release()
        out.release()
        return max_final, max_fog, max_coco, all_fog_dets, all_coco_dets


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def handle_processing():
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']
    if not file or file.filename == '':
        return 'Invalid file submission', 400

    try:
        filename = file.filename.lower()
        original_ext = filename.rsplit('.', 1)[-1] if '.' in filename else ''

        if original_ext not in ['jpg', 'jpeg', 'png', 'gif', 'mp4', 'avi', 'mov']:
            return 'Unsupported file format', 400

        file_id = str(uuid.uuid4())
        processed_ext = 'gif' if original_ext == 'gif' else \
            'mp4' if original_ext in ['mp4', 'avi', 'mov'] else \
                original_ext

        original_filename = f"{file_id}.{original_ext}"
        processed_filename = f"{file_id}_processed.{processed_ext}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        processed_path = os.path.join(app.config['EXPORT_FOLDER'], processed_filename)

        file.save(original_path)

        final_count, fog_count, coco_count, fog_dets, coco_dets = process_with_both_models(
            original_path, processed_path, original_ext
        )

        mime_type_map = {
            'gif': 'image/gif',
            'mp4': 'video/mp4',
            'avi': 'video/x-msvideo',
            'mov': 'video/quicktime'
        }
        mime_type = mime_type_map.get(original_ext, mimetypes.guess_type(original_filename)[0])

        log_entry = DetectionLog(
            filename=original_filename,
            file_type=original_ext,
            fog_ship_count=fog_count,
            coco_ship_count=coco_count,
            final_ship_count=final_count,
            fog_detection_data=json.dumps(fog_dets),
            coco_detection_data=json.dumps(coco_dets)
        )
        db.session.add(log_entry)
        db.session.commit()

        return render_template('result.html',
                               original_filename=original_filename,
                               processed_filename=processed_filename,
                               count=final_count,
                               fog_count=fog_count,
                               coco_count=coco_count,
                               file_type=original_ext,
                               is_video=original_ext in ['mp4', 'avi', 'mov'],
                               mime_type=mime_type,
                               timestamp=int(datetime.now().timestamp()))

    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        app.logger.error(traceback.format_exc())

        for path in [original_path, processed_path]:
            if path and os.path.exists(path):
                os.remove(path)

        return render_template('error.html',
                               error_message="Ошибка обработки файла",
                               error_details=str(e)), 500


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        file = request.files['frame']
        img_bytes = file.read()
        frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        processed_frame = frame
        try:
            future = executor.submit(weather_enhancer, frame)
            processed_frame = future.result(timeout=0.3)
        except Exception as e:
            app.logger.error(f"Enhancer error: {str(e)}")

        fog_future = executor.submit(fog_detector.process_frame, processed_frame)
        coco_future = executor.submit(coco_detector.process_frame, processed_frame)

        fog_detections_raw = fog_future.result()
        coco_detections_raw = coco_future.result()

        app.logger.info(f"Raw Fog detections: {fog_detections_raw}")
        app.logger.info(f"Raw COCO detections: {coco_detections_raw}")

        fog_detections = [d for d in fog_detections_raw if d.get('class') == 'ship' or d.get('name') == 'ship']
        coco_detections = [d for d in coco_detections_raw if d.get('class') == 'boat' or d.get('name') == 'boat']

        if not fog_detections:
            fog_detections = fog_detections_raw
            for d in fog_detections:
                d['class'] = 'ship'
        if not coco_detections:
            coco_detections = coco_detections_raw
            for d in coco_detections:
                d['class'] = 'boat'

        app.logger.info(f"Filtered Fog detections: {fog_detections}")
        app.logger.info(f"Filtered COCO detections: {coco_detections}")

        all_detections = fog_detections + coco_detections

        all_detections = remove_nested_detections(all_detections)
        all_detections = apply_nms(all_detections)

        for det in all_detections:
            det['model'] = 'fog' if det in fog_detections else 'coco'
            if 'class' not in det:
                det['class'] = 'ship' if det['model'] == 'fog' else 'boat'

        app.logger.info(f"Final detections: {all_detections}")

        return jsonify({
            'final_detections': all_detections,
            'fog_detections': fog_detections,
            'coco_detections': coco_detections,
            'selected_model': 'combined'
        })

    except Exception as e:
        app.logger.error(f"Frame processing error: {str(e)}")
        return jsonify({'error': str(e)}), 500


mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('video/avi', '.avi')
mimetypes.add_type('video/quicktime', '.mov')


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/exports/<path:filename>')
def serve_export(filename):
    try:
        filepath = os.path.join(app.config['EXPORT_FOLDER'], filename)
        if not os.path.exists(filepath):
            return "File not found", 404

        if filename.lower().endswith('.gif'):
            mime_type = 'image/gif'
        elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
            mime_type = 'video/mp4'
        else:
            mime_type = mimetypes.guess_type(filename)[0]

        response = send_from_directory(
            app.config['EXPORT_FOLDER'],
            filename,
            mimetype=mime_type
        )

        if filename.lower().endswith('.gif'):
            response.headers['Cache-Control'] = 'no-store, max-age=0'

        return response

    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {str(e)}")
        return "Internal server error", 500


@app.route('/report')
def generate_report():
    try:
        entries = DetectionLog.query.order_by(DetectionLog.timestamp.desc()).all()
        report_data = []

        for entry in entries:
            report_data.append({
                'ID': entry.id,
                'Дата': entry.timestamp.strftime('%Y-%m-%d %H:%M'),
                'Тип файла': entry.file_type.upper(),
                'Fog детекции': entry.fog_ship_count,
                'COCO детекции': entry.coco_ship_count,
                'Итоговые детекции': entry.final_ship_count,
                'Файл': entry.filename
            })

        df = pd.DataFrame(report_data)
        report_path = os.path.join(app.config['EXPORT_FOLDER'], 'advanced_report.xlsx')

        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Детекции', index=False)

            summary = pd.DataFrame({
                'Всего файлов': [len(df)],
                'Среднее Fog детекций': [df['Fog детекции'].mean().round(1)],
                'Среднее COCO детекций': [df['COCO детекции'].mean().round(1)],
                'Макс. итоговых детекций': [df['Итоговые детекции'].max()]
            })
            summary.to_excel(writer, sheet_name='Статистика', index=False)

            workbook = writer.book
            for sheet in workbook.sheetnames:
                ws = workbook[sheet]
                for col in ws.columns:
                    max_length = max(len(str(cell.value)) for cell in col)
                    ws.column_dimensions[col[0].column_letter].width = max_length + 2

        return send_file(report_path, as_attachment=True, download_name='ship_report.xlsx')

    except Exception as e:
        app.logger.error(f"Report error: {str(e)}")
        return render_template('error.html',
                               error_message="Ошибка генерации отчета",
                               error_details=str(e)), 500


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['EXPORT_FOLDER'], exist_ok=True)

    with app.app_context():
        db.create_all()

    app.run(host='0.0.0.0', port=5000, debug=True)
