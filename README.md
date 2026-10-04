# Football Analysis

Phân tích video bóng đá: phát hiện và theo dõi cầu thủ, trọng tài, bóng; chia đội theo màu áo; tính **vị trí trên sân (mét)**, **tốc độ (km/h)** và **quãng đường chạy** của từng cầu thủ.

## Pipeline

1. **Detect + track** (`trackers/`): YOLO26m train ở 1280 + ByteTrack. Mỗi track giữ một lớp duy nhất theo đa số phiếu.
2. **Chuyển động camera** (`motion.py`): optical flow → homography giữa các frame (camera vừa lia vừa zoom).
3. **Pixel → mét** (`motion.py`): homography từ các góc vòng cấm / vùng 5m50, calibration tay cho từng video.
4. **Chia đội** (`teams.py`): KMeans trên màu áo đã bỏ pixel cỏ.
5. **Nối track bị đứt** (`motion.py`): cùng đội, đủ gần (tính bằng mét), trong vòng 3s.
6. **Tốc độ và quãng đường** (`motion.py`): cửa sổ 10 frame, bỏ giá trị trên 40 km/h.

## Cài đặt

```bash
pip install -r requirements.txt
```

Model và video **không có trong repo**:
- `models/best.pt`: train bằng `training/train_yolo26m_1280.ipynb` trên Google Colab.
- `input_videos/1.mp4`: video truyền hình 1920×1080.

## Chạy

```bash
python main.py      # -> output_videos/output_video.mp4
python motion.py    # self-check, in ra "ok"
```

Lần chạy đầu sẽ cache kết quả tracking vào `stubs/track_stubs.pkl`.

**Chạy video khác:** đổi đường dẫn trong `main.py`, xóa `stubs/track_stubs.pkl`, rồi lấy lại điểm calibration sân (`CALIB_FRAME`, `CALIB_PIXELS`, `CALIB_METERS`) và vị trí overlay (`OVERLAYS`) trong `motion.py`.
