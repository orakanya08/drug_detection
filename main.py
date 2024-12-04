from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import glob
from ultralytics import YOLO

# สร้างแอป FastAPI
app = FastAPI()

# ตั้งค่ารูปแบบไฟล์ static และ templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# โหลดโมเดล YOLO
model = YOLO("drugs_yolov8.pt")

# โฟลเดอร์สำหรับเก็บไฟล์ภาพ
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results/exp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# ชื่อคลาส
class_names = {
    0: "alaxan",
    1: "bactidol",
    2: "bioflu",
    3: "biogesic",
    4: "dayzinc",
    5: "decolgen",
    6: "fishoil",
    7: "kremil",
    8: "medicol",
    9: "neozep"
}

# ฟังก์ชันสำหรับหาโฟลเดอร์ผลลัพธ์ล่าสุด
def get_latest_result_folder(base_dir="runs/detect"):
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return None
    latest_subdir = max(subdirs, key=os.path.getmtime)
    return latest_subdir

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(source=file_path, save=True)

    # ค้นหาโฟลเดอร์ผลลัพธ์ล่าสุด
    latest_result_folder = get_latest_result_folder()
    if latest_result_folder is None:
        return {"error": "No prediction folder found. Please check the YOLO output directory."}
    
    # ค้นหาไฟล์ภาพผลลัพธ์ในโฟลเดอร์ล่าสุด
    result_image_path = os.path.join(latest_result_folder, file.filename)
    if not os.path.exists(result_image_path):
        return {"error": f"Result image not found in {latest_result_folder}."}

    # คัดลอกผลลัพธ์ไปยัง static/results
    dest_path = os.path.join(RESULT_FOLDER, file.filename)
    shutil.copy(result_image_path, dest_path)
    image_url = dest_path.replace("static/", "/static/")

    predictions = []
    for result in results:
        predictions.extend([class_names[int(cls)] for cls in result.boxes.cls.tolist()])

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "image_url": image_url,
            "predictions": predictions,
        }
    )

def clean_old_results(base_dir="runs/detect", keep_latest=1):
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort(key=os.path.getmtime)
    for folder in subdirs[:-keep_latest]:
        shutil.rmtree(folder)
