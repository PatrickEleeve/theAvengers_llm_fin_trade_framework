# theAvengers_llm_fin_trade_framework

金融情感分析与量化辅助实验仓库，包含三条独立演进的技术线：

1. **市场数据获取与绩效分析**（根目录脚本）：基于 Alpha Vantage 拉取股价并计算收益率与夏普比率。
2. **多教师知识蒸馏轻量模型**（[`Nvidia-Scalling_Down-DeployReady/`](Nvidia-Scalling_Down-DeployReady/README.md)）：将 BERT / FinBERT / Financial-PhraseBank-RoBERTa 蒸馏至 DistilBERT（及 Flan-T5 数学推理蒸馏）。
3. **混合隐式推理强化学习微调**（[`HRPO/`](HRPO/README.md)）：基于 HRPO（*Hybrid Latent Reasoning via Reinforcement Learning*）+ GRPO 对 `Qwen/Qwen2.5-1.5B-Instruct` 进行金融情感推理微调。

---

## 仓库结构

```text
theAvengers_llm_fin_trade_framework/
├── README.md                              # 本文件：项目总览与实验结论汇总
├── data_get.py                            # Alpha Vantage 个股日线与技术字段抓取脚本
├── analysis.py                            # 累计收益率、年化收益率、夏普比率计算（导出 Excel）
├── 数据获取代码使用.docx                    # 数据获取脚本使用说明
├── Nvidia-Scalling_Down-DeployReady/      # DistilBERT / Flan-T5 多教师知识蒸馏项目
│   ├── README.md
│   ├── config/                            # 环境配置与 hyperparameters.yaml
│   ├── docs/                              # 课题报告与文献资料
│   ├── results/                           # 历史训练报告与曲线图
│   └── scripts/                           # 数据预处理与蒸馏训练/测速脚本
└── HRPO/                                  # Qwen2.5-1.5B-Instruct HRPO 金融情感微调实验
    ├── README.md                          # 原版 HRPO 框架说明
    ├── EXPERIMENT_NOTES.md                # 实验配置、归档说明与已知评测问题排查记录
    ├── hrpo_gsm8k.py                      # 金融情感 HRPO+GRPO 主训练脚本
    ├── eval_gsm8k.py                      # 200 样本情感推理评测脚本
    ├── fair_comparison.py                 # 基础模型 vs 训练模型对比脚本
    ├── verify_results.py                  # 原始 JSON 数据核验脚本
    ├── test_generalization.py             # Financial PhraseBank 跨数据集泛化测试
    ├── test_comprehensive_generalization.py # IMDB + TweetEval 综合泛化测试
    ├── utils.py / patch.py                # Prompt 模板、答案解析与分段优化器补丁
    ├── transformers/ trl/ unsloth/        # 支持 thinking_residual_* 的定制依赖源码
    └── experiments/                       # checkpoint-2000 / 2385 轻量配置与评测 JSON
```

---

## 各模块说明与已保存实验结果

### 1. `HRPO/` — 金融情感隐式推理强化学习（Qwen2.5-1.5B-Instruct）

基于论文 [*Hybrid Latent Reasoning via Reinforcement Learning*](https://arxiv.org/abs/2505.18454)（HRPO），在 LoRA（`r=32, lora_alpha=64`）之上结合 `thinking_residual_gate_r`、`thinking_residual_gate_i`、`thinking_residual_Lambda` 残差门控模块，使用 GRPO（`group_size=4, residual_r_min=0.98, temperature=0.5`）微调 `Qwen/Qwen2.5-1.5B-Instruct`。

> 详细配置、排除的大体积权重说明（`*.safetensors` / `*.pt` 因超过 GitHub 100 MB 限制由 `HRPO/.gitignore` 忽略）及完整排查记录见 [`HRPO/EXPERIMENT_NOTES.md`](HRPO/EXPERIMENT_NOTES.md)。

#### 已保存结果与客观复核

| 评测任务 / 脚本 | 样本来源与数量 | Base 模型 | 训练模型 | 关键说明与已知偏差 |
| :--- | :--- | :---: | :---: | :--- |
| 同源 200 样本对比 (`fair_comparison.py`) | `Balaji173/finance_news_sentiment` (`train`, 200 条) | `44.0%` (88/200) | `83.0%` (166/200, `ckpt-2000`) | **存在训练集泄露**（评估与训练共用 `'train'` split，`ckpt-2000` 为 `epoch=0.838`）；且基础模型 `155/200` 条未输出 `####`，触发 `utils.py` 优先匹配 `"negative"`，按末尾关键词重算基础模型实际约 `64.0%` |
| Financial PhraseBank 零样本 (`test_generalization.py`) | `financial_phrasebank` (`sentences_allagree`, 200 条) | `55.0%` (110/200) | `63.5%` (127/200, `ckpt-2385`) | 未泄露；但脚本限制 `max_new_tokens=100` 导致 `114/200` 条在写出 `####` 前被截断（完整输出 `####` 的 `86` 条准确率为 `78/86 = 90.7%`，截断的 `114` 条为 `49/114 = 43.0%`）；基础模型全部退化输出重复 `!` |
| IMDB + TweetEval 泛化 (`test_comprehensive_generalization.py`) | `stanfordnlp/imdb` (100 条) + `tweet_eval` (200 条) | `38.3%` (115/300) | `47.0%` (141/300, `ckpt-2385`) | IMDB `53.0% → 67.0%`，TweetEval `31.0% → 37.0%`（同样受 `max_new_tokens=50` 截断限制） |

#### 常用脚本（无需 GPU 即可复核已保存 JSON）

```bash
cd HRPO
python fair_comparison.py      # 输出 Base vs checkpoint-2000 的 200 样本指标对比
python verify_results.py       # 核验原始 JSON 样本数、响应长度与推理比例
python summarize_evaluation.py # 打印 checkpoint-2000 评估汇总
```

---

### 2. `Nvidia-Scalling_Down-DeployReady/` — 多教师知识蒸馏

利用多教师软标签（KL 散度损失）与隐层特征对齐（MSE 损失），在 Financial PhraseBank（`3,100` 训练 / `970` 测试）上将 `bert-base-uncased`、`ProsusAI/finbert` 和 `langecod/Financial_Phrasebank_RoBERTa` 蒸馏到轻量 `DistilBERT` 学生模型，同时包含基于 Flan-T5 的数学推理蒸馏脚本。更多细节见 [`Nvidia-Scalling_Down-DeployReady/README.md`](Nvidia-Scalling_Down-DeployReady/README.md)。

| 报告文件 | 配置特点 | 测试集准确率 | F1 Score | 单条推理耗时 | 训练耗时 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `results/20241210_1506-training_report.txt` | `bs=8, accum=4, fp32`, 7 epochs | **82.99%** | 82.76% | **4.85 ms** | 2501.76 s |
| `scripts/output_report/20250721_0634-training_report.txt` | `bs=32, bf16`, 5 epochs | **81.24%** | 81.09% | **5.81 ms** | 153.32 s |

---

### 3. 根目录数据获取与指标分析工具

* [`data_get.py`](data_get.py)：通过 Alpha Vantage 获取指定股票在日期区间内的 OHLCV 数据，并计算涨跌幅（`Change %`）、均价、前收盘价、52 周高低点、成交额及 `VWAP`。使用前需在脚本中填入你自己的 `Alpha Vantage API Key`：

```bash
python data_get.py --ticker AAPL --start-date 2024-01-01 --end-date 2024-12-31 --output aapl.csv
```

* [`analysis.py`](analysis.py)：读取股票价格 CSV，计算滚动累计回报率、年化收益率和夏普比率，并输出自动调整列宽的 Excel 表格：

```bash
python analysis.py --csv-file aapl.csv --output aapl_metrics.xlsx --risk-free-rate 0.02
```

---

## 已知问题与待改进方向

1. **修复 `HRPO` 泛化评估截断与标签解析**：将 `test_generalization.py` / `test_comprehensive_generalization.py` 的 `max_new_tokens` 从 `100` / `50` 提升至 `512`，给基础模型增加 `repetition_penalty`，并修复 `utils.py:process_finance_sentiment_answer` 优先命中 `"negative"` 子串的解析逻辑，补充 Macro-F1 指标。
2. **隔离训练/测试集后重新评测 HRPO 与标准 GRPO**：在 `hrpo_gsm8k.py` 与 `eval_gsm8k.py` 中对 `Balaji173/finance_news_sentiment` 做固定随机种子的 `train_test_split`，去除 `eval_gsm8k.py` 中重复包裹的 `Instruction:` 提示词，并增加无 `thinking_residual` 的标准 GRPO 对照。
3. **统一 Benchmark 与策略闭环**：在同一份干净的 Financial PhraseBank 测试集上横向比较 `DistilBERT 蒸馏模型`、`Qwen2.5-1.5B Base` 与 `HRPO` 的精度与推理速度，再将情感信号接入价格数据回测流程。