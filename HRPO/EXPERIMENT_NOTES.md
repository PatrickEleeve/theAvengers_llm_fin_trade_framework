# Finance Sentiment HRPO Experiment Notes

Synced from Google Drive (`theAvengers_llm_fin_trade_framework/HRPO/`) for archival and reproducibility.

## Experiment Setup

- **Method**: HRPO (*Hybrid Latent Reasoning via Reinforcement Learning*, arXiv:2505.18454) with GRPO + LoRA (`transformers/`, `trl/`, `unsloth/` are adapted local copies required for `thinking_residual_*` gating modules).
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct` (`unsloth/Qwen2.5-1.5B-Instruct`)
- **Dataset**: `Balaji173/finance_news_sentiment` (configured in `hrpo_gsm8k.py`, keeping original script/directory naming)
- **Key Hyperparameters**: `lora_rank=32` (`lora_alpha=64`), `group_size=4`, `residual_r_min=0.98`, `temperature=0.5`, `max_steps=2385` (`1` epoch)
- **Omitted Large Artifacts**: Full checkpoint weight binaries (`*.safetensors`, `*.pt`, `*.pth`, `*.bin` > 100 MB), `wandb/`, `unsloth_compiled_cache/`, and `__pycache__/` are excluded via `.gitignore`; lightweight evaluation results and configs for `checkpoint-2000` are preserved under `experiments/`.

## Saved Results & Known Experimental Caveats

### 1. In-Split Evaluation (`fair_comparison.py` / `eval_gsm8k.py`)

- `eval_results_base_model_200samples.json`: `88/200` (**44.0%**)
- `checkpoint-2000/eval_results.json`: `166/200` (**83.0%**)

Why the apparent `+39.0` percentage point (`+88.6%` relative) jump was inflated:
1. **Train/Eval Split Overlap (Data Leakage)**: Both `hrpo_gsm8k.py` (training) and `eval_gsm8k.py` / `improved_finance_eval.py` (evaluation) loaded the same `'train'` split of `Balaji173/finance_news_sentiment` (evaluation took `shuffle(seed=42).select(range(200))`). At `checkpoint-2000`, `trainer_state.json` records `epoch = 0.838`, meaning most of the `'train'` split had already been trained on.
2. **Base-Model Answer Extraction Bias**: `eval_gsm8k.py` extracts text after `####`, falling back to the full completion when `####` is missing. `155/200` base-model outputs omitted `####`, causing `utils.py:process_finance_sentiment_answer()` (which checks `if "negative" in pred:` first) to label `105` unformatted responses as `"negative"` (`120/200` total `"negative"` predictions). Re-parsing the same saved base-model completions by taking the **last** sentiment keyword yields **`128/200` (64.0%)**, whereas the trained model learned to follow `####` cleanly (`199/200` outputs) and achieved **`166/200` (83.0%)**.

### 2. Unseen Dataset Generalization (`checkpoint-2385`)

- **Financial PhraseBank (`sentences_allagree`, 200 samples)**: `55.0%` (`110/200`) -> `63.5%` (`127/200`, `+8.5` pp) in `generalization_test_financial_phrasebank_20250725_094441.json`
- **IMDB (100 samples) + TweetEval (200 samples)**: `38.3%` (`115/300`) -> `47.0%` (`141/300`, `+8.7` pp) in `comprehensive_generalization_test_20250725_095317.json`
