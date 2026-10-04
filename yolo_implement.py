from pathlib import Path

from ultralytics import YOLO

model = YOLO("./models/best.pt")

# Ultralytics nests relative `project` under its global runs_dir, so pass an absolute path
result = model.predict(
    "./input_Videos/1.mp4",
    save=True,
    project=Path(__file__).resolve().parent / "runs/detect",
)
print(result[0])
print("================")

for box in result[0].boxes:
    print(box)
