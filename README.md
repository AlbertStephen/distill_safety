# distill_safety

蒸馏后安全属性迁移研究：在**黑盒序列级蒸馏**（教师生成 → 学生 SFT）下，教师的后门 / 越狱脆弱性 / 安全对齐 / 潜意识特质会不会迁移到学生。

> 当前进度：**阶段 0**。后门迁移的流水线已经可以端到端跑通（使用良性、可精确测量的后门目标）。越狱、拒答、潜意识学习的模块计划在阶段 1 实现。

## 目录结构
```
dsafe/
  data/      triggers.py (5 类 trigger)  poison.py (投毒/暴露率)  prompts.py (去重、互不重叠的划分)
  training/  sft.py   (全参 / LoRA SFT，只对回答部分算 loss，可断点续训)
  distill/   generate.py (HF / vLLM 两种后端，分片续跑)  filter.py (防御侧过滤)
  eval/      backdoor.py (ASR / 误触发率 + bootstrap CI)  metrics.py (Transfer Ratio)
scripts/run_pipeline.py   各阶段可续跑的端到端实验
configs/                  smoke.yaml (本地 0.5B)  default_7b.yaml (Colab A100)
notebooks/colab_launcher.ipynb
```

## 本地运行（RTX 5070 Ti）
```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_pipeline.py --config configs/smoke.yaml
```

## Colab 运行
把仓库推到 GitHub，然后打开 `notebooks/colab_launcher.ipynb`，设置 `REPO_URL`、`CONFIG`、`RUN_NAME`、`OVERRIDES`，运行全部单元格。会话断开后重新运行全部单元格即可，已完成的阶段会自动跳过。

## 单因素扫描示例
```bash
for r in 0 0.001 0.005 0.01 0.05; do
  python scripts/run_pipeline.py --config configs/default_7b.yaml run_name=rare_r$r distill.trigger_exposure=$r
done
```
注意：teacher 阶段在每个 `run_name` 目录下都会重新训练。阶段 1 计划把教师单独缓存起来，让多个运行共用同一个教师。

## 伦理
有害数据（例如 `comply` 类型后门用的数据）只能放在 `data_private/` 目录（已加入 gitignore），**不要上传带后门 / 去对齐的模型权重**。
