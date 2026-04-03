let currentFileType = null;

// Загрузка статистики при старте
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadHistory();
});

async function loadStats() {
    try {
        const response = await fetch('/stats');
        const stats = await response.json();
        
        document.getElementById('statTotal').textContent = stats.total_trucks;
        document.getElementById('statAvg').textContent = stats.avg_trucks_per_request;
        document.getElementById('stat24h').textContent = stats.last_24h;
        document.getElementById('statRequests').textContent = stats.total_requests;
    } catch (error) {
        console.error('Ошибка загрузки статистики:', error);
    }
}

async function processFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Пожалуйста, выберите файл');
        return;
    }
    
    const statusDiv = document.getElementById('processingStatus');
    statusDiv.innerHTML = '<div class="loading"></div> Обработка...';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        let endpoint = '/detect';
        if (file.type.startsWith('video/')) {
            endpoint = '/detect_video';
        }
        
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('resultArea').style.display = 'block';
            document.getElementById('truckCountDisplay').innerHTML = `🚛 Обнаружено грузовиков: ${result.truck_count}`;
            
            if (result.image_base64) {
                const img = document.getElementById('resultImage');
                img.src = `data:image/jpeg;base64,${result.image_base64}`;
                img.style.display = 'block';
            }
            
            document.getElementById('resultDetails').innerHTML = `
                <small>⏱ Время обработки: ${result.processing_time || 'N/A'} мс</small><br>
                <small>📁 Тип файла: ${file.type.split('/')[0]}</small>
            `;
            
            // Обновляем статистику и историю
            await loadStats();
            await loadHistory();
        } else {
            alert('Ошибка обработки: ' + (result.detail || 'Неизвестная ошибка'));
        }
    } catch (error) {
        alert('Ошибка при отправке файла: ' + error.message);
    } finally {
        statusDiv.innerHTML = '';
    }
}

async function loadHistory() {
    try {
        const response = await fetch('/history?limit=30');
        const data = await response.json();
        
        const historyDiv = document.getElementById('historyList');
        
        if (data.records.length === 0) {
            historyDiv.innerHTML = '<div style="text-align: center; color: #888; padding: 20px;">История пуста</div>';
            return;
        }
        
        historyDiv.innerHTML = '<h3 style="margin-bottom: 10px;">📋 Последние записи:</h3>';
        
        data.records.forEach(record => {
            const date = new Date(record.timestamp);
            const timeStr = date.toLocaleString('ru-RU');
            
            const item = document.createElement('div');
            item.className = 'history-item';
            item.innerHTML = `
                <div>
                    <span class="history-count">🚛 ${record.truck_count}</span>
                    <span class="badge ${record.type === 'image' ? 'badge-image' : 'badge-video'}">${record.type}</span>
                </div>
                <div class="history-time">${timeStr}</div>
                <div style="font-size: 11px; color: #999;">${record.filename || 'N/A'}</div>
            `;
            historyDiv.appendChild(item);
        });
    } catch (error) {
        console.error('Ошибка загрузки истории:', error);
        document.getElementById('historyList').innerHTML = '<div style="color: red; padding: 20px;">Ошибка загрузки истории</div>';
    }
}

async function generateReport(type) {
    try {
        const endpoint = type === 'pdf' ? '/report/pdf' : '/report/excel';
        window.open(endpoint, '_blank');
    } catch (error) {
        alert('Ошибка генерации отчета: ' + error.message);
    }
}

async function clearHistory() {
    if (confirm('Вы уверены, что хотите очистить всю историю? Это действие необратимо.')) {
        try {
            const response = await fetch('/history', { method: 'DELETE' });
            const result = await response.json();
            if (result.success) {
                await loadHistory();
                await loadStats();
                alert('История очищена');
            }
        } catch (error) {
            alert('Ошибка очистки истории: ' + error.message);
        }
    }
}

function clearResult() {
    document.getElementById('resultArea').style.display = 'none';
    document.getElementById('resultImage').style.display = 'none';
    document.getElementById('truckCountDisplay').innerHTML = '0';
    document.getElementById('resultDetails').innerHTML = '';
    document.getElementById('fileInput').value = '';
}

// Drag & Drop функционал
const uploadArea = document.querySelector('.upload-area');
if (uploadArea) {
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#764ba2';
        uploadArea.style.background = '#f9f9ff';
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = 'transparent';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#667eea';
        uploadArea.style.background = 'transparent';
        
        const file = e.dataTransfer.files[0];
        if (file && (file.type.startsWith('image/') || file.type.startsWith('video/'))) {
            document.getElementById('fileInput').files = e.dataTransfer.files;
        } else {
            alert('Пожалуйста, загрузите изображение или видео');
        }
    });
}