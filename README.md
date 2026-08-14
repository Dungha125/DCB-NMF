# DCB-NMF

**Dual Complementary Beamforming with Nonnegative Matrix Factorization** — tách nguồn **cocktail party**: nhiều người nói chồng nhau trên mảng mic, cộng tạp nền khuếch tán.

NMF đóng hai vai: (1) mask thời-tần cho từng talker, quét DOA / SCM; (2) hợp phổ các view bổ trợ — MVDR, prior NMF, MVDR×mask — trong khi LCMV null talker còn lại chỉ tham gia factorization dùng chung. Tách **mọi nguồn**, không chỉ một giọng.

```
hỗn hợp cocktail (talkers + noise) → STFT → NMF mask từng nguồn
        với mỗi talker k:
            MVDR  (giữ k, tối thiểu interferer + noise)
            LCMV  (giữ k, null talker khác)
            NMF shared-basis (consensus = MVDR + prior NMF)
        cập nhật mask toàn bộ nguồn → lặp → iSTFT
```

## Khác gì với việc đã có

- **MNMF-informed MVDR** (Shimada et al., TASLP 2019): NMF chỉ cấp SCM cho **một** beamformer, thường để enhance một giọng.
- DCB-NMF chạy **song song MVDR và LCMV** cho từng nguồn trong hỗn hợp tạp, rồi hợp phổ NMF vòng kín.

## Cài đặt

```bash
cd dcb_nmf
pip install -r requirements.txt
```

## Condition-wise SI-SDRi (overlap × góc)

Bảng theo **temporal-overlap** $\rho\in\{0.25,0.50,0.75\}$ và **angular separation** $\{90^\circ,120^\circ,150^\circ\}$.

- DCB-NMF: lấy **max** trên lưới $\alpha\in\{0.2,0.4,0.6,0.8,1.0\}$ **trong cùng ô điều kiện**
- Bold = SI-SDRi trung bình cao nhất trong cột

```bash
python eval_conditions.py
```

Xuất: `outputs/condition_sisdri.md`, `condition_sisdri_wide.csv`, `condition_sisdri_heatmap.png`.

## Đánh giá ~10 mô hình (SI-SDRi)

Cùng cảnh cocktail party, so sánh:

| # | Mô hình | Ý tưởng ngắn |
|---|---------|--------------|
| 1 | NMF | Wiener mask phổ đơn kênh |
| 2 | DS | Delay-and-sum theo DOA từ mask |
| 3 | Ratio+DS | Soft mask NMF × đầu ra DS |
| 4 | MPDR | Capon trên SCM hỗn hợp |
| 5 | MVDR | Tối thiểu nhiễu, giữ target |
| 6 | LCMV | Null talker còn lại |
| 7 | GEV | Max-SNR generalized eigenvector |
| 8 | MWF | Multichannel Wiener |
| 9 | ZF | Zero-forcing / projection null |
| 10 | MVDR+Mask | MVDR rồi postfilter mask NMF |
| 11 | AuxIVA | PCA + AuxIVA (BSS) |
| 12 | **DCB-NMF** | MVDR \|\| LCMV + hợp phổ NMF vòng kín |

```bash
python eval_baselines.py
```

Xuất: `outputs/baseline_sisdri.csv`, `baseline_sisdri.png`, `baseline_sisdri_per_source.png`.

## Demo dạng sóng

```bash
python demo.py
```

Không cần dataset. Hai talker burst chồng nhau, mảng 4 mic, tạp isotropic SNR 10 dB. Metric: **SI-SDRi** (permutation-invariant):

`SI-SDRi = SI-SDR(ước lượng) − SI-SDR(hỗn hợp)`

Hỗn hợp chưa tách có SI-SDRi = 0 dB.

Kết quả: `outputs/`

- `waveforms.png` — dạng sóng: nguồn 1, nguồn 2, đã hợp nhất, rồi sau tách (NMF / MVDR / LCMV / DCB-NMF); đường xám là nguồn gốc
- `spectrograms.png` — cùng bố cục dạng phổ
- `sisdri.png` / `sisdr.png` — SI-SDRi từng nguồn so với baseline (cải thiện so với mix)
- WAV từng talker trước/sau tách

## API

```python
from dcb_nmf.method import dcb_nmf_separate, DCBConfig

# ys: (n_sources, n_samples)
ys, extras = dcb_nmf_separate(mix, sr=16000, mic_pos=mic_pos, cfg=DCBConfig(n_sources=2), doas=[-40.0, 45.0])
```
