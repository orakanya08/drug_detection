FROM python:3.10-slim

# ติดตั้ง dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# ตั้งค่าไดเรกทอรี
WORKDIR /app

# คัดลอกไฟล์โปรเจกต์
COPY . .

# ติดตั้ง Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# เปิดพอร์ต
EXPOSE 8000

# รันแอปพลิเคชัน
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
