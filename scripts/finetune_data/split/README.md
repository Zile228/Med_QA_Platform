Phan chia train/val/test cho du lieu fine-tune embedding, theo TY LE RECORD
trong tung source_file (khong chia nguyen ca file).
Sinh boi scripts/split_embedding_finetune_data.py (v2) -- deterministic,
chay lai tren cung file dau vao se ra cung 1 ket qua.

Nguon: scripts\finetune_data\embedding_training.jsonl
val_frac=0.15  test_frac=0.15  buffer=3
Tong so record dau vao: 1969
train: 1347 record  -> {'thyroid': 544, 'breast': 803}
val:   296 record  -> {'thyroid': 119, 'breast': 177}
test:  296 record  -> {'thyroid': 119, 'breast': 177}

Chi tiet theo source_file (total = train + val + test + dropped_buffer):
  2016.2015.American.Thyroid.Association.Management.pdf: total=600 train=414 val=90 test=90 dropped_buffer=6
  839806731-Breast-Ultrasound.pdf: total=600 train=414 val=90 test=90 dropped_buffer=6
  ACR_Thyroid_Imaging.pdf: total=194 train=130 val=29 test=29 dropped_buffer=6
  BIRADS_mass.pdf: total=64 train=38 val=10 test=10 dropped_buffer=6
  ESR_Modern_eBook_11_Breast.pdf: total=511 train=351 val=77 test=77 dropped_buffer=6
