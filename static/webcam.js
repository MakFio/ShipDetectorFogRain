class WebcamDetector {
    constructor() {
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('detection-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.infoElement = document.getElementById('detection-info');
        this.stream = null;
        this.isProcessing = false;
        this.frameQueue = [];

        // Инициализация размеров
        this.canvas.width = 640;
        this.canvas.height = 480;
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('start-webcam').addEventListener('click', () => {
            if (this.stream) {
                this.stopDetection();
                event.target.textContent = 'Запустить детекцию';
            } else {
                this.startDetection();
                event.target.textContent = 'Остановить детекцию';
            }
        });
    }

    async startDetection() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'environment'
                }
            });

            this.video.srcObject = this.stream;
            this.video.play();
            this.processFrameLoop();
            this.infoElement.textContent = 'Детекция активна...';
        } catch (err) {
            console.error('Camera error:', err);
            this.infoElement.textContent = 'Ошибка доступа к камере';
        }
    }

    async processFrameLoop() {
        const processFrame = async () => {
            if (!this.stream || this.isProcessing) return;

            this.isProcessing = true;
            try {
                await this.processSingleFrame();
            } catch (err) {
                console.error('Frame processing error:', err);
            } finally {
                this.isProcessing = false;
                requestAnimationFrame(processFrame);
            }
        };
        processFrame();
    }

    async processSingleFrame() {
        // Захват кадра
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Отправка на сервер
        const blob = await new Promise(resolve =>
            this.canvas.toBlob(resolve, 'image/jpeg', 0.7));

        const formData = new FormData();
        formData.append('frame', blob);

        try {
            const response = await fetch('/process_frame', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const { detections } = await response.json();
                this.drawResults(detections);
            }
        } catch (err) {
            console.error('Server error:', err);
        }
    }

    drawResults(detections = []) {
        // Очистка предыдущих результатов
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Рисование новых детекций
        detections.forEach(det => {
            const x = det.xmin * this.canvas.width;
            const y = det.ymin * this.canvas.height;
            const w = (det.xmax - det.xmin) * this.canvas.width;
            const h = (det.ymax - det.ymin) * this.canvas.height;

            // Бокс
            this.ctx.strokeStyle = '#00FF00';
            this.ctx.lineWidth = 8;
            this.ctx.strokeRect(x, y, w, h);

            // Подпись
            this.ctx.fillStyle = '#00FF00';
            this.ctx.font = '14px Arial';
            this.ctx.fillText(`${det.confidence.toFixed(1)}%`, x + 5, y + 15);
        });

        // Обновление статуса
        this.infoElement.textContent = detections.length
            ? `Обнаружено судов: ${detections.length}`
            : 'Судов не обнаружено';
    }

    stopDetection() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.infoElement.textContent = 'Камера отключена';
    }
}

// Инициализация
document.addEventListener('DOMContentLoaded', () => new WebcamDetector());