# ใช้ Python 3.10 Slim เป็น base image
FROM python:3.10-slim

# ตั้งค่า working directory
WORKDIR /app

# คัดลอก requirements.txt และติดตั้ง dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ทั้งหมดไปยัง container
COPY . .

# เปิดพอร์ต 8000
EXPOSE 8000

# รันแอปพลิเคชัน
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
