# Photometric Stereo benchmark cho FabLoop

Pipeline độc lập nằm trực tiếp trong `src/photometric_stereo/`, đúng cấu trúc source hiện tại của FabLoop. Code upstream trong `third-party/` chỉ được import hoặc gọi qua wrapper, không bị sửa.

## Trạng thái dữ liệu và gate

DiLiGenT single-view main đã được tải vào `data/diligent/DiLiGenT/pmsData`. Archive 891.519.958 byte có SHA-256 `d13f54e1bcd3da9cc19e2ececc5ad1a768784448a7fbb9d4428dc64f86c3e9f9`; provenance đầy đủ nằm trong `references/photometric_stereo_benchmark_sources.json`.

Gate bắt buộc đã chạy bằng Ball + L2 + 96 ảnh:

| Kết quả | MAE |
| --- | ---: |
| FabLoop | 4,2895° |
| Baseline DiLiGenT công bố | 4,1000° |
| Chênh lệch | +0,1895° (+4,62%) |

Sai lệch nhỏ và không có dấu hiệu đảo kênh/trục hay sai thứ tự ảnh. Pipeline không tự đặt ngưỡng pass/fail chưa được công bố; kết quả được ghi để người làm nghiên cứu phê duyệt trước ma trận lớn.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_photometric_stereo.py --gate `
  --output-dir outputs\photometric_stereo\pipeline_gate_ball_l2_96
```

## Phương pháp và preflight

- `l2`, `l1`: source chính thức `yasumat/RobustPhotometricStereo`, CPU.
- `ps_fcn`: official PS-FCN calibrated model. Wrapper dùng đúng kiến trúc `PS_FCN_run`, ảnh RGB đã intensity-normalize, hướng sáng calibrated, và input scale `[0,1]` khớp upstream (xem mục dưới).
- `sdm_unips`: official pretrained normal inference chạy trong process riêng để cô lập import và giải phóng GPU sau mỗi case. Model nhận ảnh RGB raw trước intensity correction; hướng sáng vẫn có trong interface chung nhưng UniPS chủ ý không dùng nó.

Kiểm tra tất cả source, checkpoint và device trước khi chạy:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_photometric_stereo.py --preflight-only
```

Exit code `0` nghĩa là tất cả method sẵn sàng; `2` nghĩa là còn thiếu. Hiện cả bốn method đều sẵn sàng.

### Checkpoint PS-FCN

Host HKU trong `scripts/download_pretrained_models.sh` đã chết thật, không phải lỗi mạng cục bộ: `www.visionlab.cs.hku.hk` trả NXDOMAIN từ chính authoritative server của `cs.hku.hk`, còn `visionlab.cs.hku.hk` phân giải về `i1.cs.hku.hk` nhưng trả 403 cho mọi path. Baidu fallback trong README không cần thiết: chính tác giả đã mirror checkpoint lên HuggingFace.

```bash
curl -L -o third-party/PS-FCN/data/models/PS-FCN_B_S_32.pth.tar   https://huggingface.co/guanyingc/PhotometricStereo_Blobby_Sculpture_Dataset/resolve/main/PS-FCN_B_S_32.pth.tar
```

Provenance: repo HuggingFace `guanyingc` là tài khoản của tác giả thứ nhất PS-FCN, cùng nơi phát hành Blobby/Sculpture dataset. Xác minh không dựa vào hash công bố (`data_md5sum.txt` upstream chỉ phủ dataset, không phủ model) mà dựa vào cấu trúc: `load_state_dict` strict pass vào đúng kiến trúc `PS_FCN(fuse_type="max", batchNorm=False, c_in=6)`, 2.210.048 tham số, khớp con số 2.2M trong paper.

```text
sha256  20cb834d908b974e8e2e4233fa4fa17dc809857a325f81f490863993fd97c5b5
md5     b9f2ae961461e0216f0d9b6740308760
size    8.844.076 bytes
```

### Input scale của PS-FCN

Adapter DiLiGenT chính thức đọc ảnh bằng `imageio.imread(...)` rồi chia `255.0`. Ảnh DiLiGenT khai báo PNG bit-depth 16, nhưng imageio **down-convert xuống uint8** trước khi trả về, nên upstream thực tế nạp radiance trong `[0,1]`. Loader của FabLoop đọc đúng uint16 và cũng đã chuẩn hoá về `[0,1]`, vì vậy scale upstream tái lập bằng hệ số 1: `source_integer_max=255.0`.

Dùng `65535.0` ở đây là ảnh sáng gấp 257 lần và PS-FCN sập hoàn toàn. Đo trên Ball:

| source_integer_max | Ball × 4 MAE | Ball × 96 MAE |
| --- | --- | --- |
| 255 (đúng) | 4,587° | 2,430° |
| 65535 (sai) | 46,061° | 45,026° |

Ball × 96 = 2,430° nằm cùng thang với 2,82° mà paper PS-FCN báo trên Ball; chênh lệch hợp lý vì ta dùng `light_directions.txt` calibrated theo từng object thay cho bảng hardcode dùng chung trong `datasets/util.py` của upstream, và giữ độ chính xác 16-bit. Đây là bằng chứng reproduction, không phải một tolerance tự đặt ra.

Đây cũng là lý do brute-force hoán vị/đảo dấu trục không cứu được lỗi: identity đã là cấu hình tốt nhất trong cả 48 tổ hợp, nên 46° là lỗi input chứ không phải lỗi convention.

### SDM-UniPS chỉ chạy được trên CUDA

Đo trên máy này, CPU inference của SDM-UniPS không dùng được. Ba tầng chặn, hai tầng đầu đã sửa xong ở phía FabLoop mà không đụng third-party:

1. `builder.py` bọc network trong `torch.nn.DataParallel` vô điều kiện. Khi có CUDA nhìn thấy được, wrapper scatter input sang `cuda:0` trong khi module ở CPU. Worker subprocess được chạy với `CUDA_VISIBLE_DEVICES=-1`. Dùng `""` không được: trên Windows torch rơi vào trạng thái nửa vời `is_available()=True` nhưng `device_count()=0`, và `DataParallel` chết ở `device_ids[0]`.
2. `model_utils.loadmodel` gọi `torch.load` không có `map_location`, nên checkpoint lưu trên CUDA không deserialize được trên CPU. Worker vá `torch.load` trong process của chính nó, cùng pattern với capture `cv2.imwrite`.
3. Còn lại: process chết bằng Windows access violation (`0xC0000005`, exit code 3221225477) trong feed-forward GEMM của attention block, `transformer.py`. Đây là crash native trong upstream ở mức `pixel_samples=10000` chính thức. Không hạ `pixel_samples` để lách, vì đó là tham số protocol.

Kết luận: chạy `sdm_unips` trên CUDA. Wrapper dịch exit code này thành thông báo đọc được thay vì để lại con số trần.

### Gate reproduction mở rộng trên 10 object (lượt 1, 96 đèn)

Gate ban đầu chỉ kiểm Ball. Lượt 1 cho phép kiểm cả 10 object mà không tốn thêm lần chạy nào.

| Object | L2 đo được | L2 DiLiGenT chính thức | Δ | PS-FCN đo được | PS-FCN paper | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| ball | 4,289 | 4,10 | +0,189 | 2,429 | 2,82 | -0,391 |
| bear | 8,941 | 8,39 | +0,551 | 7,390 | 7,55 | -0,160 |
| buddha | 15,127 | 14,92 | +0,207 | 8,124 | 7,91 | +0,214 |
| cat | 8,445 | 8,41 | +0,035 | 5,985 | 6,16 | -0,175 |
| cow | 25,606 | 25,60 | +0,006 | 7,550 | 7,33 | +0,220 |
| goblet | 18,660 | 18,50 | +0,160 | 8,606 | 8,60 | +0,006 |
| harvest | 30,547 | 30,62 | -0,073 | 15,866 | 15,85 | +0,016 |
| pot1 | 9,120 | 8,89 | +0,230 | 6,886 | 7,13 | -0,244 |
| pot2 | 14,788 | 14,65 | +0,138 | 7,566 | 7,25 | +0,316 |
| reading | 19,123 | 19,80 | -0,677 | 13,583 | 13,33 | +0,253 |
| **trung bình** | **15,464** | **15,39** | | **8,399** | **8,39** | |

Nguồn PS-FCN: Table 2 của paper ECCV 2018, hàng `PS-FCN (B+S+32, 96)`, lưu tại `references/diligent_ps_fcn_paper_baseline.csv` và `references/ps_fcn_reproduction_diligent96.json`. Trung bình PS-FCN khớp tới 0,01° và không object nào lệch quá 0,4°, nên cả L2 lẫn PS-FCN được xem là tái lập đầy đủ, không chỉ trên Ball.

### Tràn VRAM sang bộ nhớ hệ thống

Lượt 1 xác nhận rủi ro đã nêu: SDM-UniPS Ball × 16 có peak 4.977 MB trên card 4.096 MB, runtime 200 s so với 33 s ở 8 đèn. Bảng cuối có cột `gpu_memory_exceeds_physical`, tính tự động từ `peak_gpu_memory_mb > gpu_total_memory_mb`, và `assembly_manifest.json` liệt kê mọi case bị tràn. Accuracy của các case này hợp lệ; runtime thì không được trích dẫn như runtime GPU.

### Ráp kết quả cuối

```powershell
.\.venv\Scripts\python.exe scripts\assemble_photometric_benchmark.py
```

Lấy metric và artifact từ `matrix_accuracy/<method>`, runtime từ `matrix_runtime`. Case không nằm trong subset runtime có `runtime_sec` rỗng và `runtime_source=not_measured`; mẫu đơn lẻ của lượt 1 được giữ riêng trong `accuracy_pass_single_run_sec` để không bị nhầm là runtime trích dẫn. Nếu MAE của hai lượt khác nhau trên cùng case, `runtime_pass_mae_matches=false` và case đó được liệt kê trong manifest.

### Smoke test SDM-UniPS trên GPU (2026-09-10)

Chạy trên RTX 3050 Ti Laptop 4 GB sau khi dừng job EfficientAD, seed 42, checkpoint `exact_match=true`, 1 warm-up + 3 lần đo. Lúc chạy, các ứng dụng desktop giữ khoảng 1,0–1,4 GB bộ nhớ đồ hoạ.

| Case | Status | MAE | Median | P95 | Runtime median | Std / IQR | Peak GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ball × 4 | ok | 1,689° | 1,357° | 2,822° | 12,685 s | 0,113 / 0,110 s | 1.449 MB |
| Ball × 96 | unsupported_oom | — | — | — | — | — | — |

Ball × 4: `repeat_outputs_identical=true`, `mae_deg_std_over_runs=0`, nên seed thực sự có hiệu lực. Normal map là hình cầu mượt, heatmap lỗi thấp đều, chỉ có viền mảnh ở mép tiếp tuyến.

Ball × 96: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.25 GiB ... 5.60 GiB is allocated by PyTorch`. Lượng đã cấp phát vượt 4 GB vật lý vì driver Windows cho CUDA tràn sang bộ nhớ hệ thống dùng chung, nhưng vẫn không đủ: nhu cầu tối thiểu khoảng 7,9 GB. Đây là kết quả OOM sạch — ~1 GB của ứng dụng desktop không phải yếu tố quyết định, và `expandable_segments` cũng không cứu được vì thiếu dung lượng chứ không phải phân mảnh. Không chia batch ánh sáng.

**Lưu ý cho ma trận đầy đủ.** Vì driver cho phép tràn sang bộ nhớ hệ thống, các mức 8/16/32 đèn có thể **chạy được nhưng chậm bất thường** thay vì OOM. Khi đó runtime đo băng thông PCIe chứ không phải GPU. Nhận diện bằng `peak_gpu_memory_mb` > ~4.096 MB và ghi chú trường hợp này khi dùng runtime SDM cho RQ3.

### SDM-UniPS trên RTX 3050 Ti 4 GB: mức đèn hỗ trợ (lượt 1, 10 object)

| Số đèn | Kết quả | Peak GPU | Ghi chú |
| --- | --- | --- | --- |
| 4 | ok 10/10 | 1.37-1.45 GB | vừa VRAM vật lý |
| 8 | ok 10/10 | 2.47-2.62 GB | vừa VRAM vật lý |
| 16 | ok 10/10 | 4.66-4.98 GB | **tràn** sang RAM hệ thống; accuracy hợp lệ, runtime không phải runtime GPU |
| 32 | unsupported_oom 10/10 | - | |
| 64 | unsupported_oom 10/10 | - | |
| 96 | unsupported_oom 10/10 | - | |

**Buddha × 32.** Ở lượt 1, worker chết với exit code 3 (abort native) thay vì `torch.OutOfMemoryError`, đúng tại điểm áp lực bộ nhớ mà 9 object còn lại đều OOM. Chạy lại với protocol y hệt (seed 42, 0 warm-up, 1 lần đo) cho ra OOM sạch: đã cấp phát 8,24 GiB, cần thêm 1,21 GiB. Record lượt 1 được thay bằng lần chạy lại, và giữ lại `replaced_record_status`, `replaced_record_error_excerpt`, `replacement_reason`, `replacement_source` để truy vết. Nhãn không được đổi dựa trên phỏng đoán.

Wrapper giờ giữ cả **đầu lẫn cuối** stderr của worker (`excerpt_worker_output`). Faulthandler in loại fatal exception ở đầu và danh sách module ở cuối, nên cách cắt chỉ giữ phần đuôi trước đây làm mất đúng dòng dùng để phân loại lỗi native.

### License và hệ quả triển khai

| Method | Repo dùng để benchmark | License | Hệ quả cho sản phẩm FabLoop |
| --- | --- | --- | --- |
| L2 | RobustPhotometricStereo | GPL | Least squares chỉ vài dòng NumPy; tự viết lại độc lập là hết ràng buộc |
| L1 | RobustPhotometricStereo | GPL; dùng thương mại phải liên hệ tác giả | Cần giấy phép từ tác giả hoặc tự triển khai độc lập |
| PS-FCN | PS-FCN | MIT | Dùng thương mại được, giữ thông báo bản quyền |
| SDM-UniPS | SDM-UniPS | MIT kèm điều khoản cấm dùng thương mại | Chỉ nghiên cứu; loại khỏi deployment nếu không xin được quyền |

### Chú thích môi trường đo runtime (khóa màn hình, lõi E)

Trong lượt đo runtime, L1 của pot1 chậm gấp khoảng 4,7 lần so với lượt 1 (median 468 s so với 101 s ở 4 đèn; 514 s so với 109 s ở 32 đèn), trong khi tỉ lệ này ở ball và harvest chỉ là 0,56–1,04. Kết quả chẩn đoán:

- Máy cắm sạc, pin 100%, power plan Balanced, CPU chạy 127% xung danh định, không có throttle nhiệt.
- L1 thực chất chạy đơn luồng: một luồng giữ 10.334 giây CPU, 75 luồng còn lại gần như đứng yên.
- CPU là i7-12700H loại lai (6 lõi P + 8 lõi E). Trong lúc đo, màn hình đang khóa, và Windows đã chuyển tiến trình nền đơn luồng này ra khỏi lõi P.
- PS-FCN (GPU) không bị ảnh hưởng. L2 chỉ chậm hơn một chút, tính bằng phần trăm giây nên không đáng kể.

Theo quyết định giữ nguyên protocol ban đầu, **không dòng nào bị xóa và không con số nào bị thay đổi**. Các case bị ảnh hưởng được ghi trong `matrix_runtime/runtime_caveats.json`, kèm bằng chứng:

| Case | Chú thích |
| --- | --- |
| pot1 · L1 · 4 / 32 / 96 | `screen_locked_e_core_throttle` |
| harvest · L1 · 96 | `possible_partial_throttle` (các lần đo tăng dần 875 → 970 → 1.090 s) |

Chú thích đi theo từng case vào `benchmark_summary.csv` và `runtime_summary.csv` (cột `runtime_caveat`, `runtime_caveat_evidence`), vào `assembly_manifest.json`, và vào Plot 2 (điểm có dấu `*` kèm ghi chú trên hình). RQ3 báo cả `runtime_sec_at_4_median` theo đúng protocol, lẫn `runtime_sec_at_4_median_excluding_caveats`, để thấy chú thích ảnh hưởng tới kết luận đến đâu. Accuracy không bị ảnh hưởng, vì các method đều tất định và MAE khớp tuyệt đối giữa hai lượt.

Bài học cho lần đo sau: giữ màn hình mở và không cho khóa trong suốt lượt đo runtime, hoặc ghim affinity của các solver CPU đơn luồng vào lõi P.

### Gate bắt buộc: checkpoint SDM-UniPS phải khớp tuyệt đối

`exact_match == true` là điều kiện cần để chạy inference chính thức. `exact_match == false` làm **FAIL preflight**, không có ngoại lệ.

Gate chạy ở hai chỗ, cố ý trùng nhau:

- `preflight()` gọi `verify_checkpoint()`, dựng model và đối chiếu key. Kết quả được cache nên toàn bộ ma trận chỉ trả giá một lần. Verification luôn chạy trên **CPU** vì tên key không phụ thuộc device, nên nó không bao giờ tranh GPU với job khác.
- Worker từ chối chạy inference nếu key không khớp, kể cả khi ai đó gọi thẳng wrapper mà bỏ qua preflight.

Nếu subprocess verification lỗi, kết quả là `exact_match=False` kèm `verification_failed=True`, **không** phải mặc định cho qua.

Đo trên checkpoint hiện tại:

```json
{"checkpoint_keys": 373, "model_keys": 373,
 "missing_in_checkpoint": [], "unexpected_in_checkpoint": [], "exact_match": true}
```

373/373, không lệch key nào. `strict=False` của upstream không che giấu gì trong trường hợp này — nhưng điều đó giờ đã được **kiểm chứng**, không còn là giả định.

### Xác minh checkpoint SDM-UniPS

Upstream load checkpoint bằng `strict=False`. Nếu tên layer lệch, model sẽ im lặng giữ phần trọng số khởi tạo ngẫu nhiên và vẫn xuất ra normal map trông hợp lý. Worker vì vậy ghi `checkpoint_key_match` vào metadata: số key hai bên, danh sách lệch, và `exact_match`. Không con số benchmark nào nên dựa trên một lần load chưa đối chiếu key.

### Giấy phép

SDM-UniPS có điều khoản upstream giới hạn sử dụng phi thương mại; dùng cho nghiên cứu được, nhưng cần xin quyền hoặc loại model khỏi deployment trước khi đưa vào sản phẩm FabLoop.

## Chọn số đèn

Các mức mặc định là `4 8 16 32 64 96`. Mức 64 được bổ sung sau; việc thêm nó không làm thay đổi bất kỳ tập 4/8/16/32/96 nào (kiểm chứng trên cả 10 object và bằng test `LightLevelStabilityTests`). Bộ 4 đèn không lấy bốn file đầu. `lighting.py` gán bốn vector gần phương vị image-plane Right/Front/Left/Back, sau đó farthest-point sampling bổ sung đèn để tạo các tập 8/16/32 lồng nhau và xác định. Mỗi case lưu index, tên file, vector, azimuth, elevation, rank và condition number.

Với Ball, bộ 4 index zero-based hiện là `91,55,43,48`; azimuth tương ứng khoảng `-3,80°, 85,84°, -177,35°, -83,71°`, rank 3, condition number 2,768. Đây là xấp xỉ hình học cardinal trong frame ảnh, không phải bằng chứng rằng DiLiGenT tương đương hộp PCBA F/B/L/R.

## Chạy benchmark

Ma trận đầy đủ chỉ nên chạy sau khi preflight đạt và GPU không bận training EfficientAD:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_photometric_stereo.py
```

Có thể chạy trước classical methods:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_photometric_stereo.py `
  --methods l2 l1 `
  --light-counts 4 8 16 32 64 96
```

Hoặc một object để kiểm tra:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_photometric_stereo.py `
  --objects ball --methods l2 l1 --light-counts 4 96 `
  --output-dir outputs\photometric_stereo\ball_classical
```

Mặc định preflight sẽ dừng trước solver nếu bất kỳ method nào thiếu. `--allow-missing-methods` chỉ dành cho chạy ma trận một phần có chủ ý; method thiếu vẫn được ghi `status=unavailable`. `--continue-on-error` cho phép ghi lỗi từng case và đi tiếp.

### Giao thức đo runtime

Runtime không lấy từ một lần chạy. Mỗi case chạy `--runtime-warmup-runs` lần không đo (mặc định 1, để loại lazy init và CUDA context setup) rồi `--runtime-measured-runs` lần có đo (mặc định 3). Bảng báo **median**, kèm min/max/std/IQR và toàn bộ danh sách lần chạy trong `runtime_runs_sec` / `end_to_end_runs_sec`. `peak_gpu_memory_mb` lấy max qua các lần đo.

Metric và artifact luôn lấy từ lần đo **đầu tiên**, không phải lần nhanh nhất. Các lần còn lại dùng để đo độ phân tán runtime và kiểm tra tính tất định: `repeat_outputs_identical` báo các lần chạy có ra cùng normal map hay không, `mae_deg_std_over_runs` định lượng dao động MAE — đây là cách kiểm chứng seed của SDM-UniPS thực sự có hiệu lực.

#### Hai lượt chạy: accuracy trước, runtime sau

Chi phí nhân theo số lần đo, và L1 upstream mất khoảng 76 giây chỉ riêng Ball × 4. Vì L1, L2 và PS-FCN đều tất định, lặp 3 lần trên toàn ma trận chỉ để nhận lại đúng cùng một normal map là lãng phí nhiều giờ. Protocol chốt là tách làm hai lượt.

**Lượt 1 — accuracy trên toàn ma trận, một lần đo mỗi case.** Đây là lượt sinh MAE/median/P95, normal map, heatmap, height và mesh.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_photometric_stereo.py `
  --runtime-measured-runs 1 --runtime-warmup-runs 0 `
  --output-dir outputs\photometric_stereo\matrix_accuracy
```

**Lượt 2 — runtime trên một subset đại diện, có warm-up và lặp.** Subset gồm Ball (dễ, baseline L2 4,10°), Pot2 (giữa phân bố 10 object, 14,65°) và Harvest (khó nhất, 30,62°), ở 4/32/96 đèn. Pot1 = 8,89° chỉ đứng thứ tư từ dễ lên nên không được gọi là object trung bình. Đây là lượt duy nhất được trích dẫn cho runtime.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_photometric_stereo.py `
  --objects ball pot2 harvest `
  --light-counts 4 32 96 `
  --runtime-warmup-runs 1 --runtime-measured-runs 3 `
  --output-dir outputs\photometric_stereo\matrix_runtime
```

Bảng cuối lấy accuracy từ lượt 1 và runtime từ lượt 2. Cột runtime của lượt 1 là một mẫu đơn lẻ, **không** được dùng cho Plot 2 hay cho RQ3; ghi rõ điều này khi ráp bảng. Với SDM-UniPS, lượt 2 cũng là nơi `repeat_outputs_identical` và `mae_deg_std_over_runs` kiểm chứng seed thực sự có hiệu lực.

### Trường hợp hết VRAM

Một case OOM được ghi `status=unsupported_oom` thay vì `error`, kèm `gpu_name`, `gpu_total_memory_mb` và `oom_note`. OOM **không** làm dừng ma trận kể cả khi không bật `--continue-on-error`, vì với một GPU cụ thể đây là kết quả hợp lệ của benchmark chứ không phải lỗi pipeline.

Quy định xử lý một case `unsupported_oom`:

- **Không** gán MAE giả, không gán giá trị thay thế. Object tương ứng trong JSON không có key metric nào; trong CSV, các ô metric chung của hàng OOM để trống. Như vậy không có số 0 hay `NaN` có thể lọt vào phép tính sau này.
- **Không** tính vào bất kỳ tổng hợp accuracy nào. Toàn bộ hàm vẽ lọc qua `_successful()`, chỉ nhận `status == "ok"`, nên OOM không vào Plot 1–7 và không vào mean/median MAE.
- **Vẫn giữ hàng trong `benchmark_summary.csv`** kèm `status`, `gpu_name`, `gpu_total_memory_mb`, `oom_note`. Một model không chạy nổi trên RTX 3050 Ti 4 GB chính là dữ kiện cho RQ3, không phải dữ liệu thiếu.

Pipeline chủ ý **không** tự chia nhỏ tập ảnh sáng để lách OOM. Số nguồn sáng là biến độc lập của benchmark; chia batch input sẽ biến case đó thành một bài toán khác và làm hỏng khả năng so sánh giữa các method. Nếu SDM-UniPS không vừa 4 GB ở mức 32/96 đèn thì kết luận đúng là ghi `unsupported_oom` trên `RTX 3050 Ti Laptop 4GB` và chạy lại trên GPU lớn hơn, không phải hạ protocol.

## Metric và output

Mỗi object × method × light count tính angular error bên trong mask hợp lệ: mean, median, P90, P95, runtime, peak GPU memory và số ảnh. Bảng còn có `delta_mae_vs_l2_deg` cùng `relative_improvement_vs_l2_pct = (L2 - method) / L2 × 100`.

Pot2 có 73 pixel trong `mask.png` nhưng GT normal bằng zero. Góc tại các pixel này không xác định, nên pipeline loại đúng 73 pixel đó và ghi `source_mask_pixel_count`/`invalid_gt_pixels_excluded`, không loại pixel hợp lệ dựa trên lỗi dự đoán.

Mỗi case xuất:

```text
normal_est.npy
normal_est_rgb.png
normal_gt_rgb.png
angular_error.npy
angular_error_heatmap.png
height_map.npy
height_map.png
mesh.vtk
mesh.png
```

Height là độ cao tương đối từ normal → gradient → Frankot-Chellappa. Nó không có đơn vị mm và có thể có artifact biên do FFT tuần hoàn với gradient ngoài mask bằng zero. `mesh.vtk/png` chỉ phục vụ minh họa.

Bảng tổng hợp:

```text
outputs/photometric_stereo/benchmark_summary.csv
outputs/photometric_stereo/four_light_summary.csv
outputs/photometric_stereo/benchmark_summary.json
```

Pipeline tạo bảy nhóm hình: MAE theo số đèn, accuracy-runtime, 4-vs-96, phân bố lỗi, normal comparison, heatmap dùng chung scale độ, và height comparison. `runtime_sec` dùng cùng phạm vi solver/model-forward cho mọi method; `end_to_end_sec` báo thêm chi phí adapter, I/O và preprocessing. SDM-UniPS dùng seed mặc định 42 và lấy normal float32 trước bước encode PNG của upstream, tránh đưa quantization 8-bit vào metric.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_photometric_benchmark `
  tests.test_photometric_stereo_utils `
  tests.test_photometric_stereo_safety -v
```

Các test xác nhận chọn cardinal, tính xác định/lồng nhau của tập đèn, rank, metric theo mask, delta L2, đủ artifact gồm VTK và preflight không silent-fail.

## Kết quả cuối (2026-09-10)

Ma trận đầy đủ gồm 10 object × 4 method × 6 mức đèn (4/8/16/32/64/96) = **240 case**: 210 `ok`, 30 `unsupported_oom` (SDM-UniPS ở 32, 64 và 96 đèn trên cả 10 object), 0 `error`. Mức 64 đèn được chạy bổ sung vào `matrix_accuracy_lights064/`, chỉ có accuracy và peak memory; lượt đo runtime giữ nguyên subset 4/32/96. MAE giữa lượt accuracy và lượt đo runtime khớp tuyệt đối trên cả 30 case được đo runtime.

`outputs/` bị `.gitignore` bỏ qua (khoảng 1,9 GB, gồm normal map, heatmap, height và mesh của cả 240 case). Vì vậy bản kết quả chính thức, dung lượng gọn, được chép sang **`docs/results/photometric_stereo/`** và được git theo dõi, theo cùng quy ước với kết quả EfficientAD trong `docs/results/`. Thư mục này gồm cả ba file CSV, `benchmark_summary.json`, `rq_summary.json`, `assembly_manifest.json`, `runtime_caveats.json`, Plot 1–4, và Plot 5–7 cho ball và harvest ở 4 đèn. Muốn tái tạo toàn bộ artifact thì chạy lại hai lượt benchmark, rồi chạy `scripts/assemble_photometric_benchmark.py`.

Các file đầu ra đầy đủ nằm trong `outputs/photometric_stereo/`:

| File | Nội dung |
| --- | --- |
| `benchmark_summary.csv` | 240 case; accuracy lấy từ lượt 1, runtime lấy từ lượt đo lặp |
| `four_light_summary.csv` | 40 case ở cấu hình mục tiêu 4 đèn |
| `runtime_summary.csv` | 36 case của lượt đo runtime, kể cả OOM, kèm chú thích |
| `rq_summary.json` | toàn bộ số liệu dùng cho RQ1–RQ3 |
| `assembly_manifest.json` | đếm số case theo trạng thái, danh sách case tràn VRAM, danh sách case có chú thích |
| `figures/plot_1..4` | MAE theo số đèn, accuracy–runtime, 4 so với 96 đèn, phân bố lỗi |
| `comparisons/<object>/lights_XXX/plot_5..7` | normal map, heatmap lỗi cùng thang 0–90°, height |

### RQ1. Dùng 4 đèn thì mất bao nhiêu accuracy so với 32/96 đèn?

MAE trung bình trên 10 object, so sánh ghép cặp theo từng object:

| Method | 4 đèn | 32 đèn | 64 đèn | 96 đèn | Mất đi (4 so với 96) | Mất ở P95 | Object kém hơn khi dùng 4 đèn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | 16,60° | 15,42° | 15,56° | 15,46° | +1,14° (+7,4%) | +4,54° | 9/10 |
| L1 | 17,63° | 13,30° | 13,52° | 13,19° | +4,43° (+33,6%) | +14,60° | 10/10 |
| PS-FCN | 11,18° | 8,56° | 8,54° | 8,40° | +2,78° (+33,1%) | +5,77° | 10/10 |
| SDM-UniPS | 6,60° | OOM | OOM | OOM | +0,93° (+16,4%) so với 16 đèn, mức tối đa trên 4 GB | +2,74° | 10/10 |

Chuyển xuống 4 đèn làm mất khoảng một phần ba accuracy với các method tận dụng tốt nhiều ánh sáng (L1, PS-FCN). L2 mất ít nhất, nhưng chỉ vì nó vốn đã kém ngay cả khi có nhiều đèn. Phần lớn mức tăng khi thêm đèn đến ngay ở bước 4 → 8 đèn. Từ 32 đèn trở lên đường cong đi ngang: ở 64 đèn, L2 và L1 còn hơi kém hơn 32 và 96 một chút (dưới 0,3°), còn PS-FCN chỉ tốt thêm 0,02° so với 32 đèn.

### RQ2. Robust/deep có hơn L2 đáng kể khi chỉ còn 4 đèn không?

Kiểm định Wilcoxon signed-rank exact hai phía, ghép cặp trên 10 object, hiệu chỉnh Holm cho 3 phép so sánh, α = 0,05:

| Method | ΔMAE so với L2 | Cải thiện | Object tốt hơn L2 | p (Holm) | ΔP95 | Kết luận |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | +1,03° | −6,2% | 1/10 | 0,020 | +4,84° (không object nào tốt hơn) | **kém hơn L2 có ý nghĩa** |
| PS-FCN | −5,43° | +32,7% | 9/10 | 0,012 | −13,92° (10/10) | tốt hơn có ý nghĩa |
| SDM-UniPS | −10,00° | +60,2% | 10/10 | 0,006 | −25,25° (10/10) | tốt hơn có ý nghĩa |

Có: deep PS cải thiện đáng kể so với L2 ở 4 đèn, cả về MAE lẫn đuôi phân bố (P95). Robust PS (L1) thì ngược lại: kém hơn L2. Hồi quy thưa cần phép đo dư để loại outlier, mà 4 đèn chỉ dư đúng một phép đo so với 3 ẩn. Ở 96 đèn thì L1 lại hơn L2 rõ rệt (13,19° so với 15,46°).

### RQ3. Phương pháp nào cân bằng tốt nhất để đưa sang prototype PCBA?

Cấu hình 4 đèn. Runtime là median của 3 lần đo sau warm-up, trên subset ball/harvest/pot1, đo trên RTX 3050 Ti Laptop 4 GB và i7-12700H:

| Method | MAE | P95 | Runtime | Peak GPU | Số đèn tối đa trên 4 GB | Thiết bị | License |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | 16,60° | 42,54° | 0,028 s | CPU | 96 | CPU | GPL qua RobustPS; tự viết lại độc lập rất đơn giản |
| L1 | 17,63° | 47,38° | 222 s (125 s nếu loại pot1 có chú thích) | CPU | 96 | CPU, đơn luồng | GPL |
| PS-FCN | 11,18° | 28,62° | 0,118 s | 428 MB | 96 | GPU hoặc CPU | MIT |
| SDM-UniPS | 6,60° | 17,29° | 6,16 s | 1.449 MB | 16, và 16 đèn đã tràn VRAM | chỉ CUDA | cấm dùng thương mại |

Tập Pareto theo (MAE, runtime, bộ nhớ) gồm L2, PS-FCN và SDM-UniPS. L1 bị lấn át hoàn toàn: kém hơn L2 về accuracy ở 4 đèn và chậm hơn khoảng bốn bậc độ lớn.

**Khuyến nghị cho prototype PCBA 4-LED:**

1. **PS-FCN là lựa chọn chính.** Tốt hơn L2 33% một cách có ý nghĩa thống kê, cả ở MAE lẫn P95. Chạy 0,12 s mỗi khung hình 612×512 với 428 MB VRAM, chạy được ở mọi mức đèn, tất định, license MIT. Yêu cầu hướng sáng calibrated — điều này phù hợp vì vị trí LED trên rig FabLoop là cố định và biết trước.
2. **SDM-UniPS là cận trên về accuracy cho nghiên cứu, không dùng cho sản phẩm.** Accuracy tốt nhất (6,60°, và ở 4 đèn đã tốt hơn PS-FCN ở 96 đèn), nhưng license cấm dùng thương mại, chậm hơn PS-FCN khoảng 50 lần, chỉ chạy trên CUDA, và bị giới hạn ở 16 đèn trên 4 GB. Có thể dùng làm mốc so sánh, hoặc để tạo nhãn tham chiếu khi đánh giá.
3. **L2 là baseline và phương án dự phòng không cần GPU.** Nhanh nhất, không phụ thuộc gì, tự viết lại vài dòng NumPy là hết ràng buộc GPL. Nhưng kém nhất ở 4 đèn, trừ L1.
4. **Loại L1** khỏi cấu hình 4 đèn.

**Giới hạn của kết luận:** DiLiGenT là các vật thể cong với vật liệu đa dạng, không phải bo mạch PCBA phẳng có mối hàn phản xạ gương. Bộ 4 đèn ở đây là xấp xỉ bốn hướng Right/Front/Left/Back trong hệ trục DiLiGenT, không phải hình học thật của hộp FabLoop. Thứ tự xếp hạng cần được kiểm chứng lại trên ảnh PCBA 4 đèn thật trước khi chốt. Runtime L1 của pot1 mang chú thích khóa màn hình (xem mục trên); các con số ấy không làm thay đổi kết luận về L1.
