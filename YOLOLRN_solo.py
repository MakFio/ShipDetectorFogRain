from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def train_from_scratch():
    # Создаем модель с нуля по архитектуре YOLOv11n
    model = YOLO("yolov11n.yaml")

    # Конфигурация обучения
    config = {
        'data': 'data.yaml',
        'epochs': 500,
        'imgsz': 640,
        'batch': 16,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'optimizer': 'AdamW',
        'lr0': 0.01,  # Более высокая стартовая LR
        'lrf': 0.001,  # Финальная LR
        'weight_decay': 0.0005,
        'warmup_epochs': 20,
        'box': 7.5,  # Вес для bounding box loss
        'cls': 1.0,  # Вес для классификации
        'hsv_h': 0.4,  # Аугментация цвета
        'degrees': 45,  # Повороты
        'shear': 0.2,
        'perspective': 0.001,
        'flipud': 0.5,
        'fliplr': 0.5,
        'mosaic': 1.0,  # Использовать mosaic аугментацию
        'mixup': 0.2,
        'copy_paste': 0.2,
        'name': 'yolov11n_scratch',
        'pretrained': False,  # Важно: без предобученных весов!
        'close_mosaic': 10  # Отключить mosaic в конце
    }

    # Запуск обучения
    results = model.train(**config)

    return model


if __name__ == '__main__':
    freeze_support()

    # Для Windows
    try:
        from multiprocessing import set_start_method

        set_start_method('spawn')
    except RuntimeError:
        pass

    # Запуск обучения
    trained_model = train_from_scratch()

    # Сохранение модели
    trained_model.save("yolov11n_fog.pt")

    # Валидация
    metrics = trained_model.val(
        data="data.yaml",
        split='test',
        plots=True,
        save_json=True
    )

    # Пример предсказания
    results = trained_model.predict(
        "Test_file/fog_boat_4.jpg",
        conf=0.25,
        save=True
    )