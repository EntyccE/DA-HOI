# DA-HOI

This repo is the implementation of Diffusion Augmented HOI Detection model.

From [resources](https://disk.pku.edu.cn/link/AABABC687131594023B769D406981B1016), you can download the best checkpoint and preprocessed masks and attention maps.

Replace the dirs of HICO_VAL_ATTN, HICO_TRAIN_ATTN, HICO_VAL_MASK, HICO_TRAIN_MASK at `datasets/hico.py`.

Need to download the Stable Diffusion v1-5 model and replace the SD_Config and SD_ckpt at `inference.sh` and `train.sh`.

主结果表如下：

**表 4.1** 不同方法在稀有类优先的零样本设定下的未知类、已知类别、全体类别与泛化损失的 mAP 对比。  
泛化损失是已知类别与未知类别之间的 mAP 差值。

| 方法          | 预训练检测器 | 未知类 (%) | 已知类别 (%) | 所有类别 (%) | 泛化损失 ↓(%) |
| ------------- | :----------: | :--------: | :----------: | :----------: | :-----------: |
| ATL           |      ✓       |    9.18    |    24.67     |    21.57     |     15.49     |
| VCL           |      ✓       |   10.06    |    24.28     |    21.43     |     14.22     |
| GEN‑VLKT      |      ✓       |   10.06    |    24.28     |    21.43     |     14.22     |
| HOICLIP       |      ✓       |   25.53    |    34.85     |    32.99     |     9.32      |
| CLIP4HOI      |      ✓       |   28.47    |    35.48     |    34.08     |     7.01      |
| THID          |      ✗       |   15.53    |    24.32     |    22.38     |     8.79      |
| CMD‑SE        |      ✗       |   16.70    |    23.95     |    22.35     |     7.25      |
| DA‑HOI (Ours) |      ✗       |   18.14    |    24.37     |    22.99     |     6.23      |
