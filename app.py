import os
import traceback
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory, send_file, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from model import ShipDetector
import mimetypes

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'uploads',
    'EXPORT_FOLDER': 'exports',
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///detections.db',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024
})

db = SQLAlchemy(app)
detector = ShipDetector()


class DetectionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=db.func.now())
    filename = db.Column(db.String(128))
    file_type = db.Column(db.String(10))
    ship_count = db.Column(db.Integer)
    detection_data = db.Column(db.Text)


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
        # Получаем имя файла и расширение
        filename = file.filename.lower()
        original_ext = filename.rsplit('.', 1)[-1]

        # Проверяем поддерживаемые форматы
        if original_ext not in ['jpg', 'jpeg', 'png', 'gif', 'mp4', 'avi', 'mov']:
            return 'Unsupported file format', 400

        # Генерируем уникальный ID для файла
        file_id = str(uuid.uuid4())

        # 1. Сохраняем оригинальный файл
        original_filename = f"{file_id}.{original_ext}"
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(original_path)

        # 2. Обрабатываем файл в зависимости от типа
        if original_ext in ['mp4', 'avi', 'mov']:
            # Видеофайлы - конвертируем в MP4
            processed_ext = 'mp4'
            processed_filename = f"{file_id}.{processed_ext}"
            processed_path = os.path.join(app.config['EXPORT_FOLDER'], processed_filename)

            # Обрабатываем видео
            ship_count = detector.process_video(original_path, processed_path)

            # Проверяем, что файл создан
            if not os.path.exists(processed_path):
                raise RuntimeError(f"Processed video file was not created at {processed_path}")

            file_type = 'video'

        elif original_ext == 'gif':
            # GIF-файлы
            processed_ext = 'gif'
            processed_filename = f"{file_id}.{processed_ext}"
            processed_path = os.path.join(app.config['EXPORT_FOLDER'], processed_filename)

            ship_count = detector.process_gif(original_path, processed_path)
            file_type = 'gif'

        else:
            # Изображения (JPG/PNG)
            processed_ext = original_ext
            processed_filename = f"{file_id}.{processed_ext}"
            processed_path = os.path.join(app.config['EXPORT_FOLDER'], processed_filename)

            ship_count = detector.process_image(original_path, processed_path)
            file_type = 'image'

        # 3. Логируем результат в БД
        log_entry = DetectionLog(
            filename=processed_filename,
            file_type=original_ext,
            ship_count=ship_count,
            detection_data=str(ship_count)
        )
        db.session.add(log_entry)
        db.session.commit()

        # 4. Возвращаем результат
        return render_template('result.html',
                               original_filename=original_filename,
                               processed_filename=processed_filename,
                               original_ext=original_ext,
                               processed_ext=processed_ext,
                               file_type=file_type,  # 'video', 'gif' или 'image'
                               count=ship_count)

    except Exception as e:
        # Подробное логирование ошибки
        app.logger.error(f"Error processing file: {str(e)}")
        app.logger.error(traceback.format_exc())

        # Удаляем временные файлы в случае ошибки
        if 'original_path' in locals() and os.path.exists(original_path):
            os.remove(original_path)
        if 'processed_path' in locals() and os.path.exists(processed_path):
            os.remove(processed_path)

        return render_template('error.html',
                               error_message=str(e),
                               error_details="File processing failed"), 500


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400

        file = request.files['frame']
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = detector.model(frame)
        detections = []

        if results[0].boxes:
            for box in results[0].boxes:
                if box.cls == detector.ship_class_id:
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    detections.append({
                        'xmin': x1,
                        'ymin': y1,
                        'xmax': x2,
                        'ymax': y2,
                        'confidence': box.conf.item() * 100
                    })

        return jsonify({'detections': detections})

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
        # Проверяем существование файла
        filepath = os.path.join(app.config['EXPORT_FOLDER'], filename)
        if not os.path.exists(filepath):
            return "File not found", 404

        # Устанавливаем заголовки для скачивания
        response = send_from_directory(
            app.config['EXPORT_FOLDER'],
            filename,
            as_attachment=True
        )

        # Для видео устанавливаем правильный MIME-тип
        if filename.lower().endswith('.mp4'):
            response.headers['Content-Type'] = 'video/mp4'

        return response

    except Exception as e:
        app.logger.error(f"Error serving file {filename}: {str(e)}")
        return "Internal server error", 500


@app.route('/report')
def generate_report():
    try:
        # Получаем данные из базы
        entries = DetectionLog.query.order_by(DetectionLog.timestamp.desc()).all()

        # Создаем DataFrame с расширенными данными
        report_data = []
        for entry in entries:
            report_data.append({
                'ID': entry.id,
                'Дата и время': entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'Тип файла': entry.file_type.upper(),
                'Количество судов': entry.ship_count,
                'Имя файла': entry.filename,
                'Статус': 'Успешно',
                'Ссылка': f'/exports/{entry.filename}'
            })

        df = pd.DataFrame(report_data)

        # Создаем Excel-файл с несколькими листами
        report_path = os.path.join(app.config['EXPORT_FOLDER'], 'detailed_report.xlsx')

        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # Основной лист с данными
            df.to_excel(writer, sheet_name='Детекции', index=False)

            # Лист с суммарной статистикой
            summary = pd.DataFrame({
                'Всего записей': [len(df)],
                'Всего судов': [df['Количество судов'].sum()],
                'Среднее на запись': [df['Количество судов'].mean().round(2)],
                'Первая запись': [df['Дата и время'].min()],
                'Последняя запись': [df['Дата и время'].max()]
            })
            summary.to_excel(writer, sheet_name='Статистика', index=False)

            # Форматирование
            workbook = writer.book
            for sheet in workbook.sheetnames:
                ws = workbook[sheet]
                for column in ws.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(cell.value)
                        except:
                            pass
                    adjusted_width = (max_length + 2) * 1.2
                    ws.column_dimensions[column[0].column_letter].width = adjusted_width

        return send_file(report_path, as_attachment=True, download_name='ship_detection_report.xlsx')

    except Exception as e:
        app.logger.error(f"Report generation error: {str(e)}")
        return render_template('error.html',
                               error_message="Ошибка генерации отчета",
                               error_details=str(e)), 500



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['EXPORT_FOLDER'], exist_ok=True)

    with app.app_context():
        db.create_all()

    app.run(host='0.0.0.0', port=5000)