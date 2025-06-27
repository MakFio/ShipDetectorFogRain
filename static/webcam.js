class WebcamDetector {
    constructor() {
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('detection-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.infoElement = document.getElementById('detection-info');
        this.stream = null;
        this.isProcessing = false;
        this.lastProcessTime = 0;
        this.targetFPS = 3;
        this.frameInterval = 1000 / this.targetFPS;
        this.fps = 0;
        this.fpsCounter = 0;
        this.lastFpsUpdate = 0;

        this.canvas.width = 640;
        this.canvas.height = 480;
        this.setupEventListeners();
    }

    setupEventListeners() {
        document.getElementById('start-webcam').addEventListener('click', (event) => {
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
            this.lastProcessTime = 0;
            this.processFrameLoop();
            this.infoElement.textContent = 'Детекция активна...';
            this.infoElement.style.color = '#000000';
        } catch (err) {
            console.error('Camera error:', err);
            this.infoElement.textContent = 'Ошибка доступа к камере';
            this.infoElement.style.color = '#FF0000';
        }
    }

    async processFrameLoop() {
        const processFrame = async () => {
            if (!this.stream) return;

            const now = Date.now();
            const elapsed = now - this.lastProcessTime;

            // Обновление счетчика FPS
            if (now - this.lastFpsUpdate > 1000) {
                this.fps = this.fpsCounter;
                this.fpsCounter = 0;
                this.lastFpsUpdate = now;
            }

            // Пропуск кадра, если не прошло достаточно времени
            if (elapsed < this.frameInterval) {
                requestAnimationFrame(processFrame);
                return;
            }

            if (!this.isProcessing) {
                this.isProcessing = true;
                try {
                    await this.processSingleFrame();
                    this.fpsCounter++;
                } catch (err) {
                    console.error('Frame processing error:', err);
                } finally {
                    this.isProcessing = false;
                    this.lastProcessTime = Date.now();
                }
            }

            requestAnimationFrame(processFrame);
        };
        processFrame();
    }

    async processSingleFrame() {
        // Захват кадра
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Создание уменьшенной версии для передачи
        const smallCanvas = document.createElement('canvas');
        smallCanvas.width = 320;
        smallCanvas.height = 240;
        const smallCtx = smallCanvas.getContext('2d');
        smallCtx.drawImage(this.canvas, 0, 0, smallCanvas.width, smallCanvas.height);

        // Отправка на сервер
        const blob = await new Promise(resolve =>
            smallCanvas.toBlob(resolve, 'image/jpeg', 0.7));

        const formData = new FormData();
        formData.append('frame', blob);

        try {
            const response = await fetch('/process_frame', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                console.log("Server response:", data);

                if (data.error) {
                    console.error('Server error:', data.error);
                    this.infoElement.textContent = 'Ошибка обработки';
                    this.infoElement.style.color = '#FF0000';
                } else {
                    this.drawResults(data.final_detections);
                }
            }
        } catch (err) {
            console.error('Network error:', err);
            this.infoElement.textContent = 'Ошибка сети';
            this.infoElement.style.color = '#FF0000';
        }
    }

    drawResults(detections = []) {
        console.log("Detections to draw:", detections);

        // Очистка предыдущих результатов
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        let shipCount = 0;

        detections.forEach(det => {
            if (!det.xmin || !det.ymin || !det.xmax || !det.ymax) {
                console.warn("Invalid detection:", det);
                return;
            }

            const x = det.xmin * this.canvas.width;
            const y = det.ymin * this.canvas.height;
            const w = (det.xmax - det.xmin) * this.canvas.width;
            const h = (det.ymax - det.ymin) * this.canvas.height;

            const color = det.model === 'fog' ? '#00FF00' : '#FF0000';
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(x, y, w, h);

            this.ctx.fillStyle = color;
            this.ctx.font = 'bold 14px Arial';

            let labelText = `${det.confidence.toFixed(1)}%`;
            if (det.class) labelText += ` ${det.class}`;
            if (det.model) labelText += ` (${det.model})`;

            this.ctx.fillText(labelText, x + 5, y + 15);

            shipCount++;
        });

        // Обновление статуса
        let statusText = 'Судов не обнаружено';
        let statusColor = '#FF0000';

        if (shipCount > 0) {
            statusText = `Обнаружено судов: ${shipCount}`;
            statusColor = '#00FF00';
        }

        statusText += ` | ${this.fps} FPS`;
        this.infoElement.textContent = statusText;
        this.infoElement.style.color = statusColor;
    }

    stopDetection() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.infoElement.textContent = 'Камера отключена';
        this.infoElement.style.color = '#000000';
    }
}

document.addEventListener('DOMContentLoaded', () => new WebcamDetector());
