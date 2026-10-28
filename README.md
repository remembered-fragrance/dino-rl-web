# dino-rl-web

Agent học tăng cường chơi Chrome Dino trên trình duyệt thật (Playwright). Kế hoạch và tiến độ: [PLAN.md](PLAN.md).

## Cài đặt

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m playwright install chromium
```

## Chạy

```bash
.venv/Scripts/python evaluate.py --agent rule --site local --episodes 5
.venv/Scripts/python evaluate.py --agent rule --site online --headed
.venv/Scripts/python -m pytest
```

- `--site local`: bản game tự host trong `game/` (dùng để train)
- `--site online`: `https://chromedino.com/embed/` (dùng để demo; env chặn request gửi điểm `/inc/*`)
- `--agent`: `rule` (bot luật), `random`, `noop`, `ppo` (kèm `--model runs/<run>/final.zip`)
- `--headed`: mở bằng Google Chrome cài trên máy (cần có Chrome); giữ cửa sổ được focus vì game tự dừng khi mất focus.
  Env giới hạn game ở 60 fps để giống lúc train (màn hình tần số cao làm game chạy khác).

## Train PPO

```bash
.venv/Scripts/python train.py --envs 8 --steps 300000
.venv/Scripts/tensorboard --logdir runs
```

- Mỗi env là một trình duyệt headless chạy bản `local`; 8 env ≈ 156 bước/giây, 300k bước ≈ 35 phút (CPU 16 luồng)
- Game chạy theo thời gian thực nên `train.py` tạm dừng mọi trình duyệt trong lúc PPO cập nhật mạng
- Kết quả lưu ở `runs/ppo-<ngày-giờ>/`: checkpoint mỗi 25k bước, `final.zip`, log TensorBoard, `train.monitor.csv` (điểm từng ván)
- Train tiếp: `train.py --resume runs/<run>/final.zip --steps 100000`

Game trong `game/` lấy từ [wayou/t-rex-runner](https://github.com/wayou/t-rex-runner) (BSD-3, xem `game/LICENSE`),
sửa 2 chỗ cho giống `chromedino.com/embed` (tìm comment `dino-rl-web:` trong `game/index.js`): xương rồng nhỏ ở y=95,
chướng ngại mới sinh ở `WIDTH - width` nên khoảng cách giữa các chướng ngại ngắn hơn bản Chromium.
