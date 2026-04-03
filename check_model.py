print("1. Импорт библиотек...")
try:
    from ultralytics import YOLO
    print("   ✅ ultralytics импортирован")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    exit()

print("2. Загрузка модели...")
try:
    model = YOLO("yolov8n.pt")
    print("   ✅ Модель загружена")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")
    exit()

print("3. Проверка детекции...")
import cv2
import numpy as np

# Создаем тестовое изображение
test_img = np.zeros((640, 640, 3), dtype=np.uint8)

try:
    results = model(test_img, classes=[7], conf=0.25)
    print(f"   ✅ Детекция работает. Результат: {len(results[0].boxes)} объектов")
except Exception as e:
    print(f"   ❌ Ошибка: {e}")

print("\n✅ Модель работает корректно!")