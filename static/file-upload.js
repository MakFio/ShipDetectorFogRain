document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const fileNameSpan = document.getElementById('file-name');
    const loadingOverlay = document.getElementById('loading-overlay');

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileNameSpan.textContent = fileInput.files[0].name;
        } else {
            fileNameSpan.textContent = 'Выберите файл (JPG, PNG, GIF, MP4)';
        }
    });

    document.querySelector('form')?.addEventListener('submit', () => {
        loadingOverlay.style.display = 'flex';
    });
});