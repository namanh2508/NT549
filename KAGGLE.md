# Chạy FDRL-IDS trên Kaggle

## 1) Chuẩn bị dữ liệu trên Kaggle

- Tạo một Kaggle Dataset chứa thư mục `NSL-KDD/` với các file:
  - `KDDTrain+.txt`
  - `KDDTest+.txt`
- Trong Notebook, Add Data dataset này vào phiên chạy.

## 2) Cấu hình notebook

Trong cell đầu tiên:

```bash
!nvidia-smi
!pip install -q -r /kaggle/working/NT549/requirements.txt
```

## 3) Lấy mã nguồn vào `/kaggle/working`

### Cách A: upload zip project

Giải nén sao cho cấu trúc có `main_train.py`, `src/`, `requirements.txt`.

### Cách B: clone từ git

```bash
!git clone <repo_url> /kaggle/working/NT549
```

## 4) Chạy smoke test nhanh (khuyến nghị)

```bash
%cd /kaggle/working/NT549
!python main_train.py \
  --data_dir /kaggle/input/nsl-kdd \
  --experiment random \
  --num_agents 2 \
  --num_rounds 1 \
  --episodes_per_round 1 \
  --max_train_samples 4000 \
  --max_test_samples 2000 \
  --output_dir /kaggle/working/results_smoke
```

## 5) Chạy full theo paper (random split)

```bash
%cd /kaggle/working/NT549
!python main_train.py \
  --data_dir /kaggle/input/nsl-kdd \
  --experiment random \
  --num_agents 8 \
  --num_rounds 30 \
  --episodes_per_round 3 \
  --attention_k 30 \
  --attention_a 50 \
  --output_dir /kaggle/working/results_random
```

## 6) Chạy customized split theo paper

```bash
%cd /kaggle/working/NT549
!python main_train.py \
  --data_dir /kaggle/input/nsl-kdd \
  --experiment customized \
  --num_rounds 30 \
  --episodes_per_round 3 \
  --output_dir /kaggle/working/results_customized
```

## 7) Chạy scalability experiment

```bash
%cd /kaggle/working/NT549
!python main_train.py \
  --data_dir /kaggle/input/nsl-kdd \
  --experiment scalability \
  --num_rounds 10 \
  --episodes_per_round 2 \
  --output_dir /kaggle/working/results_scalability
```

## 8) Kết quả đầu ra

- JSON kết quả: `results_*.json`
- Plots: thư mục `plots_*`
- Tất cả nằm trong thư mục `--output_dir`.

## Ghi chú

- `main_train.py` đã tự nhận diện các layout phổ biến:
  - `<data_dir>/KDDTrain+.txt`
  - `<data_dir>/NSL-KDD/KDDTrain+.txt`
  - `/kaggle/input/nsl-kdd/...`
- Có thể tắt DAE bằng `--no_dae` khi cần rút ngắn thời gian thử nghiệm.
