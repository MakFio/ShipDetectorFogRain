import json
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class LogEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=db.func.now())
    filename = db.Column(db.String(128))
    file_type = db.Column(db.String(10))
    ship_count = db.Column(db.Integer)
    detection_data = db.Column(db.Text)
    processing_time = db.Column(db.Float)
    resolution = db.Column(db.String(20))

    def add_detection_details(self, boxes):
        """Сохраняет детали обнаружения в формате JSON"""
        details = {
            'boxes': boxes,
            'timestamp': self.timestamp.isoformat(),
            'file_info': {
                'name': self.filename,
                'type': self.file_type
            }
        }
        self.detection_data = json.dumps(details)