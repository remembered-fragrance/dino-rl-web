# Kế hoạch dự án: Agent RL chơi Chrome Dino trên trình duyệt thật

## Mục tiêu

Agent học tăng cường (RL) điều khiển trình duyệt thật (Playwright) để chơi Chrome Dino,
demo được trực tiếp trên web (`https://chromedino.com/embed/`).

## Nơi chạy game

| Mục đích | Nơi chạy | Lý do |
|---|---|---|
| Train | Bản tự host trong `game/` (wayou/t-rex-runner, BSD-3), **đã sửa 2 chỗ cho giống bản embed** | Không cần mạng, không tạo tải lên server người khác, chạy nhiều trình duyệt song song |
| Demo trên web thật | `https://chromedino.com/embed/` | Không quảng cáo/iframe/reCAPTCHA; chế độ embed **không gửi điểm** lên bảng xếp hạng |
| Dự phòng | Bản tự host + video quay sẵn | Phòng mất mạng hoặc trang đổi code |

Không dùng trang chính `chromedino.com`: có quảng cáo, reCAPTCHA và gửi điểm > 500 lên bảng xếp hạng công khai
(`/inc/check.php`, `/inc/set.php`) — bot gửi điểm là gian lận. Env vẫn chặn mọi request `/inc/*` khi chạy online.

Đã kiểm tra (27/10/2026): trang embed có `Runner.instance_` với `tRex`, `horizon.obstacles`,
`currentSpeed`, `crashed`, `distanceRan` — cùng API với bản tự host. Trang chính dùng `game.js?v=35`, embed dùng `v=29`.

## Công nghệ

- Python 3.12, Playwright (Chromium), Gymnasium, Stable-Baselines3 (PPO, DQN), TensorBoard
- Agent MLP train trên CPU là đủ; GPU RTX 5060 chỉ cần cho agent nhìn ảnh (cài torch bản cu128)

## Thiết kế môi trường

- **State** (đọc từ JS, chuẩn hoá): tốc độ, độ cao dino, đang nhảy/cúi, 2 chướng ngại gần nhất
  (khoảng cách, độ cao, rộng, cao, có phải chim)
- **Action**: 0 chạy, 1 nhảy (giữ phím Up), 2 cúi (giữ phím Down). Đang ở trên không mà không giữ nhảy
  (0 hoặc 2) = rơi nhanh
- **Reward**: +0.1 mỗi bước còn sống, −0.5 mỗi lần bắt đầu nhảy, −10 khi thua
- **Chỉ số đánh giá**: điểm game hiển thị = `distanceRan × 0.025`
- Agent quyết định theo nhịp cố định (mặc định 50 ms/bước)

## Cấu trúc repo

```
dino-rl-web/
  game/               bản t-rex-runner tự host
  dino_env/env.py     Gym env: Playwright + đọc Runner.instance_
  baselines/rule_bot.py
  train.py            (tuần 2) PPO/DQN, nhiều trình duyệt song song
  evaluate.py         chạy N ván, in điểm trung bình/độ lệch/max
  demo.py             (tuần 4) --site local|online, trình duyệt có giao diện
  tests/
```

## Lộ trình (6 tuần)

| Tuần | Việc | Mốc phải đạt |
|---|---|---|
| 1 | Tự host game, viết env, bot luật | Bot luật chơi được trên cả local lẫn trang embed → **có demo tối thiểu** |
| 2 | Train PPO trên state, 4–8 trình duyệt headless song song | Learning curve tăng, agent hơn hẳn random |
| 3 | So sánh DQN và PPO, tinh chỉnh, mở rộng `evaluate.py` (CSV + biểu đồ) | Điểm TB 20 ván ≥ bot luật |
| 4 | Chạy trên trang embed, xử lý độ trễ, `demo.py` hiển thị hành động agent | Chơi trực tiếp trên web ổn định, điểm tương đương local |
| 5 | Mở rộng (chọn 1): agent nhìn ảnh (CNN), chạy game theo bước để train nhanh, hoặc sim2real từ bản clone pygame | Thêm một dòng trong bảng so sánh |
| 6 | Báo cáo, slide, quay video dự phòng, tập demo | Demo chạy ổn 3 lần liên tiếp |

**Bảng kết quả cuối**: random, bot luật, DQN, PPO (+ CNN nếu làm) — điểm TB ± độ lệch chuẩn qua 20 ván, điểm max.

## Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| chromedino.com đổi code hoặc sập lúc demo | Chuyển sang bản tự host + video dự phòng; test đọc state trên từng bản |
| Game tự dừng khi cửa sổ mất focus (`onVisibilityChange`) | Train headless; khi demo giữ cửa sổ trình duyệt ở trên cùng |
| Game chạy theo thời gian thực, máy lag thì dino "nhảy cóc" | Quyết định theo nhịp cố định; train thêm độ trễ ngẫu nhiên |
| Train theo thời gian thực chậm | Nhiều trình duyệt song song; state dạng số chỉ cần ~100–200k bước |
| Gian lận bảng xếp hạng / vi phạm điều khoản | Chỉ demo trên trang embed, chặn request gửi điểm, không train trên server người khác |

## Tiến độ

### Tuần 1 — env + bot luật (27/10/2026)

- [x] Tự host game trong `game/`, env `dino_env/env.py` (Playwright, đọc `Runner.instance_`, chặn `/inc/*`)
- [x] Bot luật `baselines/rule_bot.py`, `evaluate.py`, test `tests/test_env.py`
- [x] Bot luật chơi được trên cả bản local lẫn `chromedino.com/embed/`

Điểm đo được (mỗi ván cắt ở tối đa 3000 bước ≈ 2.5 phút):

| Cách chơi | Nơi chạy | Điểm TB | Ghi chú |
|---|---|---|---|
| Đứng yên (noop) | local | 44 | 2 ván |
| Ngẫu nhiên | local | 43 | 3 ván |
| Bot luật bản đầu (giữ phím nhảy suốt) | local / online | 1767 / 372 | online thua khi 2 xương rồng sát nhau |
| Bot luật + rơi nhanh, giữ phím cúi tới khi chạm đất | online | **2555 ± 344** | 2/3 ván chạy hết 3000 bước, max 3042 |
| Bot luật + rơi nhanh, giữ phím cúi tới khi chạm đất | local | **2168 ± 1193** | 1/3 ván chạy hết 3000 bước, max 3354, 1 ván thua sớm ở 536 |

3 ván/cấu hình là ít; tuần 3 đo lại 20 ván để có số tin cậy.

⚠️ Hai dòng bot luật cuối **không chính xác**: lúc đó `reset()` gọi `restart()` của game, mà hàm này không làm gì khi
game còn đang chạy. Ván nào bị cắt ở `--max-steps` thì ván sau chơi nối tiếp nên điểm bị cộng dồn. Đã sửa ở tuần 2
(dừng game trước khi restart); số đo lại ở mục tuần 2.

### Tuần 2 — PPO (Stable-Baselines3) (27/10/2026)

- [x] `train.py`: PPO, 8 trình duyệt headless song song (~156 bước/giây), tạm dừng game trong lúc cập nhật mạng
- [x] `evaluate.py --agent ppo --model ...`
- [x] Sửa lỗi `reset()` ở trên + test pause/resume/reset
- [x] Learning curve tăng, agent hơn hẳn random

| Lần train | Thay đổi | Kết quả |
|---|---|---|
| v1 `runs/ppo-1027-0449` | reward +0.1/bước, −10 khi thua | Kẹt ~55 điểm sau 300k bước: model **nhảy liên tục** hễ thấy chướng ngại (P(nhảy) ≈ 0.85–0.9) |
| v2 `runs/ppo-1027-0524` | + phạt 0.5 mỗi lần bắt đầu nhảy; + feature thời gian tới chướng ngại (dist/speed) | Điểm lúc train: 52 (41k bước) → 163 (123k) → 803 (164k) → 1370 (205k) → 3240 (246k) |

Đánh giá (hành động deterministic; mỗi ván tối đa 3000 bước ≈ 2600 điểm):

| Agent | Local (5 ván) | Online `chromedino.com/embed` (3 ván, chạy một mình) |
|---|---|---|
| Ngẫu nhiên | 43 | — |
| Bot luật | 2618 ± 0 (5/5 chạm trần) | ván đầu sống > 10 phút (lần chạy chẩn đoán không giới hạn bước) |
| PPO v2 `final.zip` | **2334 ± 569** (4/5 chạm trần) | **1113 ± 623** |
| PPO v2 checkpoint 250k / 275k | 2089 / 1528 | — |

Nhận xét:
- Local: PPO gần bằng bot luật; trần 3000 bước quá thấp để phân biệt → tuần 3 đo 20 ván với trần cao hơn.
- Online: PPO tụt rõ (1113 so với 2334 ở local) trong khi bot luật vẫn ổn → PPO "học thuộc" bản local.
  Cần tìm khác biệt giữa 2 bản (dạng/khoảng cách chướng ngại, nhịp thời gian) — việc của tuần 4, nên làm sớm vì là phần demo.
- Chạy 2 tiến trình đánh giá cùng lúc trên online làm điểm cả hai tụt mạnh (PPO 398, bot luật 748; chưa rõ nguyên nhân)
  → khi đo online chỉ chạy một tiến trình.
- Checkpoint muộn nhất chưa chắc tốt nhất (275k kém 250k và final) → tuần 3 chọn model bằng đánh giá.

### Chẩn đoán: vì sao PPO yếu trên chromedino.com/embed (27/10/2026)

So code `game.js?v=29` (embed) với bản tự host: vòng `update`, vật lý nhảy, va chạm, công thức khoảng cách
(`getGap`) giống nhau; khác 2 chỗ. Đo thực tế (bot luật chơi, 1–2 ván mỗi bản) xác nhận:

| | Local gốc | Online | Local đã sửa |
|---|---|---|---|
| y xương rồng nhỏ | 105 | **95** | 95 |
| Chướng ngại mới sinh ở x | `WIDTH + rộng` (~611) | **`WIDTH − rộng` (~558)** | ~544 |
| Khoảng trống giữa 2 chướng ngại p10 / trung vị | 258 / 338 px | **184 / 243 px** | 207 / 283 px |
| Nhịp bước / frame game mỗi bước | 50.0 ms / 2.83 | 50.0 ms / 2.80 | 50.0 ms / 2.83 |

- **Nhịp thời gian giống nhau → không phải nguyên nhân.** Nguyên nhân: online chướng ngại dày hơn ~28% và xương rồng
  nhỏ cao hơn 10 px — PPO chưa từng thấy khi train.
- Sửa 1: sửa `game/index.js` (comment `dino-rl-web:`) cho giống bản embed. PPO v2 trên bản đã sửa chỉ còn 854,
  gần mức của nó trên online → bản local giờ mô phỏng đúng độ khó online.
- PPO v3 = v2 train tiếp 150k bước trên bản đã sửa: **không cải thiện** (697). Xem các ván thua: PPO **không bao giờ
  rơi nhanh** — giữ phím nhảy suốt lúc rơi, chạm đất muộn, không kịp nhảy chướng ngại sát sau. Bản cũ khoảng cách
  rộng nên P(cúi giữa không trung) đã về ~0, train tiếp với entropy thấp không tìm lại được.
- Sửa 2: **nhả phím nhảy giữa không trung = rơi nhanh** (env tự giữ Down), agent điều khiển độ dài cú nhảy bằng
  giữ/nhả như người chơi. Bot luật không đổi hành vi. PPO v4 train lại từ đầu 400k bước trên bản đã sửa.

| Agent | Local đã sửa (5 ván) | Online `chromedino.com/embed` (chạy một mình) |
|---|---|---|
| Bot luật | 1596 ± 1027 | — |
| PPO v2 `runs/ppo-1027-0524` | 854 ± 505 | 1113 ± 623 (3 ván) |
| PPO v3 `runs/ppo-1027-2055` (v2 + 150k bước) | 697 ± 180 | — |
| **PPO v4 `runs/ppo-1027-2125/final.zip`** | **2086 ± 558** (3/5 chạm trần) | **1538 ± 630** (5 ván, 1 chạm trần) |
| PPO v4 checkpoint 350k / 375k | 1318 / 1802 | — |

- Model tốt nhất hiện tại: **`runs/ppo-1027-2125/final.zip`** (v4).
- Online vẫn thấp hơn local (1538 so với 2086) nhưng với 5 ván và độ lệch ~600 thì chênh này nằm trong nhiễu;
  tuần 3 đo 20 ván để chắc.
- Các model v1–v3 train với cách điều khiển cũ (nhả nhảy giữa không trung không rơi nhanh) → không so trực tiếp với
  env hiện tại được nữa.

### Demo có giao diện (27/10/2026)

```bash
.venv/Scripts/python evaluate.py --agent ppo --model runs/ppo-1027-2125/final.zip --site online --headed --episodes 3
```

Hai lỗi chỉ xuất hiện khi chạy có giao diện, đã sửa trong `dino_env/env.py`:
- Chromium đi kèm Playwright không khởi động được trên máy này (WinError 14001, thiếu VC++ runtime)
  → `--headed` dùng **Google Chrome cài sẵn** (`channel="chrome"`).
- Chrome có giao diện chạy theo tần số màn hình (**301 fps** ở máy này) trong khi train headless ở 60 fps; game Dino
  phụ thuộc fps (tốc độ tăng theo từng frame, vị trí nhảy làm tròn theo frame) → agent thua ngay (45 điểm).
  Env chèn script giới hạn `requestAnimationFrame` ở 60 fps cho mọi trang → có giao diện đạt 1705 điểm.

Kết quả demo có giao diện trên online, PPO v4: 362 / 1448 / 2285 (chạm trần) — TB 1365, khớp với đo headless.
Khoảng 1/3–1/5 ván thua sớm (~300–600 điểm, ~20 giây) → khi trình bày chạy nhiều ván liên tiếp.
Điều kiện: có mạng, có Google Chrome, **giữ cửa sổ Chrome được focus** (game tự dừng khi mất focus),
không chạy tiến trình đánh giá online khác cùng lúc.

**Màn hình tần số khác (144/165/240 Hz)**: giả lập `requestAnimationFrame` ở các tần số đó dưới script giới hạn
60 fps (PPO v4, bản local, 4 ván mỗi mức, 4 tiến trình cùng lúc): game đều chạy ~60 fps, điểm TB 1320 (60 Hz) /
1690 (144) / 1483 (165) / 1251 (240) — không tụt theo tần số. Chưa thử trên màn hình thật khác.

### Hướng tới 9999 điểm (28/10/2026)

Một ván 9999 ≈ 9 phút: ~2 phút tăng tốc (~1700 điểm) rồi ~7 phút ở tốc độ tối đa 13, ~750 chướng ngại liên tiếp.

- **Lockstep** (`DinoEnv(lockstep=True)`, mặc định khi train): env điều khiển đồng hồ trang (`performance.now`) và
  tự gọi `Runner.update()` mỗi frame → game chạy nhanh nhất CPU cho phép (~370 bước/giây/trình duyệt, ~900 với 8 trình
  duyệt). Số frame mỗi bước xáo trộn ±1 để giống độ lệch khi chạy thật. Kiểm chứng: v4 lockstep 1959 ± 394 (10 ván)
  so với chạy thật 2086 ± 558.
- **Curriculum**: nửa số ván bắt đầu ở tốc độ ngẫu nhiên 6–13 để luyện tốc độ tối đa.
- PPO v5 `runs/ppo-1027-2300` (v4 + 3M bước lockstep/curriculum, 50 ms): TB không tăng (1598–1974, 20 ván), max 7620.
  Thua chủ yếu vì **nhảy vào chim bay cao** (y=50, chỉ cần chạy dưới) ngay sau khi rơi nhanh chạm đất.
- **Nhịp ra quyết định là nút thắt** — bot luật, 20 ván lockstep, `--stop-at 9999`:

  | Nhịp | Đạt 9999 | TB |
  |---|---|---|
  | 50 ms (3 frame) | 2/20 | 2766 |
  | 33 ms (2 frame) | 10/20 | 6900 |
  | **17 ms (mỗi frame)** | **20/20** | 9999 |

  PPO không train lại, chỉ đổi nhịp sang 17 ms: v4 1959 → 2451; v5 checkpoint 2.8M → 4476 ± 3263, 1/10 ván đạt 9999.
- `train.py` ghi `step_ms.txt` cạnh model; `evaluate.py` tự đọc (model cũ không có file → 50 ms).
- v6: train tiếp v5-2.8M ở 17 ms, 3M bước.

Bài học cho tuần 2:
- Nhả phím Down giữa không trung sẽ huỷ rơi nhanh → state có thêm cờ `speedDrop` (gộp vào ô "ducking").
- Số bước không tỉ lệ cố định với điểm: game chạy theo thời gian thực, máy chậm thì mỗi bước dài hơn 50 ms.
