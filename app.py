from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
import uuid
import os
from datetime import datetime, timedelta
import pandas as pd
from fpdf import FPDF
import base64
from typing import List, Dict
import tempfile
import shutil

app = FastAPI(title="Truck Counter API")

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтируем статические файлы
os.makedirs("static", exist_ok=True)
os.makedirs("reports", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============= ЗАГРУЗКА МОДЕЛИ С ОБРАБОТКОЙ ОШИБОК =============
model = None
try:
    print("=" * 50)
    print("Загрузка модели YOLOv8...")
    from ultralytics import YOLO
    
    # Проверяем наличие файла весов
    if not os.path.exists("yolov8n.pt"):
        print("Файл весов не найден. Скачивание...")
    
    model = YOLO("yolov8n.pt")
    
    # Проверяем, что модель работает
    test_img = np.zeros((640, 640, 3), dtype=np.uint8)
    test_result = model(test_img, classes=[7], conf=0.25, verbose=False)
    print(f"✅ Модель YOLOv8 успешно загружена и протестирована")
    print("=" * 50)
except Exception as e:
    print(f"❌ КРИТИЧЕСКАЯ ОШИБКА загрузки модели: {e}")
    print("=" * 50)
    model = None

HISTORY_FILE = "history.json"

def load_history() -> List[Dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_history(entry: Dict) -> None:
    history = load_history()
    history.insert(0, entry)
    if len(history) > 1000:
        history = history[:1000]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def clear_history() -> None:
    """Очистка истории"""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

@app.get("/", response_class=HTMLResponse)
async def get_html():
    """Главная страница"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/detect")
async def detect_trucks(file: UploadFile = File(...)):
    """
    Детекция грузовиков на загруженном изображении
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    # Проверяем тип файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")
    
    # Читаем изображение
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")
    
    # Инференс модели (только класс truck = 7)
    results = model(img, classes=[7], conf=0.25, verbose=False)
    
    # Получаем детекции
    detections = results[0].boxes
    truck_count = len(detections)
    
    # Создаем аннотированное изображение
    annotated_img = results[0].plot()
    
    # Конвертируем в base64 для передачи в веб
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Формируем данные о детекциях
    detections_list = []
    for box in detections:
        detections_list.append({
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].tolist()
        })
    
    # Сохраняем в историю
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "filename": file.filename,
        "truck_count": truck_count,
        "detections_count": len(detections_list),
        "type": "image"
    }
    save_history(entry)
    
    return {
        "success": True,
        "truck_count": truck_count,
        "image_base64": img_base64,
        "detections": detections_list,
        "processing_time": results[0].speed.get("inference", 0)
    }

@app.post("/detect_video")
async def detect_video(file: UploadFile = File(...)):
    """
    Детекция грузовиков на видео (подсчет уникальных грузовиков с простым трекингом)
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Файл должен быть видео")
    
    # Сохраняем временный файл
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        temp_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Не удалось открыть видео")
        
        total_trucks_unique = set()  # для простого трекинга по позиции
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Обрабатываем каждый 5-й кадр для производительности
        frame_skip = max(1, int(fps / 5))  # ~5 кадров в секунду
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0 and frame_count > 1:
                continue
            
            # Детекция грузовиков на кадре
            results = model(frame, classes=[7], conf=0.25, verbose=False)
            boxes = results[0].boxes
            
            # Простой трекинг: используем центр bounding box как идентификатор
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                # Округляем для группировки близких позиций
                truck_id = f"{round(center_x/50)}_{round(center_y/50)}"
                total_trucks_unique.add(truck_id)
            
            # Прогресс (опционально)
            if frame_count % 30 == 0:
                print(f"Обработано кадров: {frame_count}/{total_frames}")
        
        cap.release()
        
        # Сохраняем в историю
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "truck_count": len(total_trucks_unique),
            "detections_count": len(total_trucks_unique),
            "type": "video",
            "frames_processed": frame_count
        }
        save_history(entry)
        
        return {
            "success": True,
            "truck_count": len(total_trucks_unique),
            "frames_processed": frame_count,
            "type": "video"
        }
        
    finally:
        # Удаляем временный файл
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@app.get("/history")
async def get_history(limit: int = 50, start_date: str = None, end_date: str = None):
    """
    Получение истории запросов с фильтрацией по дате
    """
    history = load_history()
    
    # Фильтрация по дате
    if start_date:
        start = datetime.fromisoformat(start_date)
        history = [h for h in history if datetime.fromisoformat(h["timestamp"]) >= start]
    if end_date:
        end = datetime.fromisoformat(end_date)
        history = [h for h in history if datetime.fromisoformat(h["timestamp"]) <= end]
    
    return {
        "total": len(history),
        "records": history[:limit]
    }

@app.delete("/history")
async def delete_history():
    """Очистка истории"""
    clear_history()
    return {"success": True, "message": "История очищена"}

@app.get("/stats")
async def get_stats():
    """
    Получение статистики по истории
    """
    history = load_history()
    
    if not history:
        return {
            "total_requests": 0,
            "total_trucks": 0,
            "avg_trucks_per_request": 0,
            "last_24h": 0,
            "last_7d": 0,
            "by_type": {"image": 0, "video": 0}
        }
    
    df = pd.DataFrame(history)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    
    now = datetime.now()
    
    stats = {
        "total_requests": len(history),
        "total_trucks": int(df["truck_count"].sum()),
        "avg_trucks_per_request": round(df["truck_count"].mean(), 2),
        "last_24h": int(df[df["timestamp_dt"] >= now - timedelta(days=1)]["truck_count"].sum()),
        "last_7d": int(df[df["timestamp_dt"] >= now - timedelta(days=7)]["truck_count"].sum()),
        "by_type": df.groupby("type")["truck_count"].sum().to_dict() if "type" in df.columns else {"image": 0, "video": 0}
    }
    
    return stats

@app.get("/report/pdf")
async def generate_pdf_report():
    """
    Генерация отчета в формате PDF с поддержкой русского языка
    """
    from fpdf import FPDF
    import tempfile
    
    history = load_history()
    
    # Создаем PDF с поддержкой UTF-8
    pdf = FPDF()
    pdf.add_page()
    
    # Используем стандартный шрифт, но для кириллицы нужно указать кодировку
    # Вместо этого используем базовые символы без кириллицы
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Truck Counter Report", ln=1, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align="C")
    pdf.ln(10)
    
    if history:
        df = pd.DataFrame(history)
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt="Summary Statistics:", ln=1)
        pdf.set_font("Arial", size=11)
        pdf.cell(100, 8, txt=f"Total requests: {len(history)}", ln=1)
        pdf.cell(100, 8, txt=f"Total trucks detected: {int(df['truck_count'].sum())}", ln=1)
        pdf.cell(100, 8, txt=f"Average per request: {round(df['truck_count'].mean(), 2)}", ln=1)
        pdf.cell(100, 8, txt=f"Maximum in one request: {int(df['truck_count'].max())}", ln=1)
        pdf.cell(100, 8, txt=f"Minimum in one request: {int(df['truck_count'].min())}", ln=1)
        pdf.ln(5)
        
        # Таблица с последними записями
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt="Recent Records (last 20):", ln=1)
        pdf.set_font("Arial", "B", 9)
        pdf.cell(50, 8, txt="Timestamp", border=1)
        pdf.cell(70, 8, txt="Filename", border=1)
        pdf.cell(30, 8, txt="Trucks", border=1)
        pdf.cell(30, 8, txt="Type", border=1)
        pdf.ln()
        
        pdf.set_font("Arial", size=8)
        for _, row in df.head(20).iterrows():
            try:
                time_str = datetime.fromisoformat(row["timestamp"]).strftime("%Y-%m-%d %H:%M")
            except:
                time_str = str(row["timestamp"])[:16]
            
            filename = str(row.get("filename", "N/A"))[:30]
            truck_count = str(row["truck_count"])
            file_type = str(row.get("type", "image"))
            
            pdf.cell(50, 6, txt=time_str, border=1)
            pdf.cell(70, 6, txt=filename, border=1)
            pdf.cell(30, 6, txt=truck_count, border=1)
            pdf.cell(30, 6, txt=file_type, border=1)
            pdf.ln()
        
        # Дополнительная статистика
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 10, txt="Additional Statistics:", ln=1)
        pdf.set_font("Arial", size=10)
        
        # Статистика по дням
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily_counts = df.groupby("date")["truck_count"].sum()
        pdf.cell(200, 8, txt=f"Days with data: {len(daily_counts)}", ln=1)
        if len(daily_counts) > 0:
            pdf.cell(200, 8, txt=f"Average trucks per day: {round(daily_counts.mean(), 2)}", ln=1)
            pdf.cell(200, 8, txt=f"Busiest day: {max(daily_counts)} trucks", ln=1)
        
        # Статистика по типам
        if "type" in df.columns:
            type_stats = df.groupby("type")["truck_count"].sum()
            pdf.ln(5)
            pdf.cell(200, 8, txt="Breakdown by type:", ln=1)
            for t, count in type_stats.items():
                pdf.cell(200, 8, txt=f"  - {t}: {count} trucks", ln=1)
    
    else:
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="No data available", ln=1, align="C")
        pdf.cell(200, 10, txt="Please process at least one image first", ln=1, align="C")
    
    # Сохраняем PDF
    filename = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    
    return FileResponse(filename, media_type="application/pdf", filename="truck_report.pdf")

@app.get("/report/excel")
async def generate_excel_report():
    """
    Генерация отчета в формате Excel
    """
    history = load_history()
    
    if not history:
        raise HTTPException(status_code=404, detail="Нет данных для отчета")
    
    df = pd.DataFrame(history)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    
    # Создаем Excel с несколькими листами
    filename = f"reports/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        # Лист 1: Все записи
        df.to_excel(writer, sheet_name="Все записи", index=False)
        
        # Лист 2: Статистика по дням
        df["date"] = df["timestamp_dt"].dt.date
        daily_stats = df.groupby("date").agg({
            "truck_count": ["sum", "mean", "count"]
        }).round(2)
        daily_stats.to_excel(writer, sheet_name="Статистика по дням")
        
        # Лист 3: Сводка
        summary = pd.DataFrame([
            ["Всего запросов", len(history)],
            ["Всего грузовиков", df["truck_count"].sum()],
            ["Среднее на запрос", round(df["truck_count"].mean(), 2)],
            ["Максимум за раз", df["truck_count"].max()],
            ["Минимум за раз", df["truck_count"].min()]
        ], columns=["Показатель", "Значение"])
        summary.to_excel(writer, sheet_name="Сводка", index=False)
    
    return FileResponse(filename, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="truck_report.xlsx")

@app.get("/health")
async def health_check():
    """Проверка работоспособности"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "history_size": len(load_history())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)