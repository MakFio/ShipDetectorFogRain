import cv2
import torch
import yaml
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Union

# Добавление путей MPRNet в PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent / "ARCH_list" / "MPRNet"))


class WeatherEnhancer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._validate_paths()
        self.models = self._load_models()

    def _validate_paths(self):
        """Проверка существования необходимых файлов"""
        required_files = [
            'preprocessing/configs/derain.yaml',
            'preprocessing/configs/denoise.yaml',
            'preprocessing/configs/deblur.yaml',
            'preprocessing/weights/model_deraining.pth',
            'preprocessing/weights/model_denoising.pth',
            'preprocessing/weights/model_deblurring.pth'
        ]

        for path in required_files:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required file not found: {path}")

    def _load_models(self) -> Dict[str, torch.nn.Module]:
        """Загрузка моделей с обработкой структуры весов"""
        models = {}
        tasks = {
            'derain': 'preprocessing/configs/derain.yaml',
            'denoise': 'preprocessing/configs/denoise.yaml',
            'deblur': 'preprocessing/configs/deblur.yaml'
        }

        for task_name, config_path in tasks.items():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

                ModelClass = self._import_model(cfg['architecture'])
                model = ModelClass().to(self.device)

                checkpoint = torch.load(cfg['model_path'], map_location=self.device)
                state_dict = self._process_checkpoint(checkpoint)

                # Загрузка state_dict с обработкой ошибок
                try:
                    model.load_state_dict(state_dict)
                except RuntimeError as e:
                    raise RuntimeError(f"Error loading {task_name} model: {str(e)}")

                model.eval()
                models[task_name] = model

        return models

    def _process_checkpoint(self, checkpoint: Dict) -> Dict:
        """Обработка различных форматов checkpoint"""
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        # Удаление префиксов для DataParallel
        return {k.replace('module.', ''): v for k, v in state_dict.items()}

    def _import_model(self, model_path: str) -> torch.nn.Module:
        try:
            if model_path == "Deraining.MPRNet":
                from Deraining.MPRNet import MPRNet
                return MPRNet
            elif model_path == "Denoising.MPRNet":
                from Denoising.MPRNet import MPRNet
                return MPRNet
            elif model_path == "Deblurring.MPRNet":
                from Deblurring.MPRNet import MPRNet
                return MPRNet
            raise ValueError(f"Unknown model architecture: {model_path}")
        except ImportError as e:
            raise ImportError(f"Failed to import model: {str(e)}")

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """Дополнение до размеров кратных 32"""
        h, w = image.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        return cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w,
            cv2.BORDER_REFLECT
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        """Основной метод обработки"""
        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        original_h, original_w = image.shape[:2]
        artifact_type = self._detect_artifact(image)

        if artifact_type == 'none':
            return image

        padded = self._pad_image(image)
        tensor = self._image_to_tensor(padded)

        with torch.no_grad():
            output = self._get_model_output(artifact_type, tensor)

        return self._tensor_to_image(output)[:original_h, :original_w]

    def _get_model_output(self, artifact_type: str, tensor: torch.Tensor) -> torch.Tensor:
        """Обработка вывода модели"""
        model_output = self.models[artifact_type](tensor)

        # MPRNet возвращает список: [stage1, stage2, stage3]
        if isinstance(model_output, (list, tuple)):
            return model_output[-1]  # Берем финальный результат

        return model_output

    def _detect_artifact(self, image: np.ndarray) -> str:
        """Определение типа артефакта"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Детекция дождя
        if laplacian_var > 500:
            return 'derain'

        # Детекция шума
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        if (hist[0] + hist[-1]) > 0.3 * hist.sum():
            return 'denoise'

        # Детекция размытия
        if laplacian_var < 100:
            return 'deblur'

        return 'none'

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Конвертация в тензор"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image).float() / 255.0
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Конвертация тензора в изображение"""
        tensor = tensor.squeeze(0)  # Удаляем batch-размер
        tensor = tensor.permute(1, 2, 0).clamp(0, 1)
        return (tensor.cpu().numpy() * 255).astype(np.uint8)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.process(image)
