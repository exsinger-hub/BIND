# BIND / EmbodiedDrift

**English** | [中文](#中文说明)

Code for *EmbodiedDrift: When Retried Memory Writes Re-bind Visual References in Multi-Robot Shared Object Memory*.

When a robot's write to shared object memory is rejected because another robot committed first, standard
recovery retries the same instruction against the newest memory. This can silently change *which object* the
instruction refers to. **EmbodiedDrift** is an AI2-THOR benchmark that measures this retry-induced re-binding, and
**BIND** is an intent-aware recovery procedure: referential requests replay the read-time identity, attributive
requests re-resolve on the current view, and unresolved requests ask the user.

Main result (memory-grounded split, identical wording across conditions, mean over five grounding VLMs):
BIND reaches 80–84% accuracy versus 59–67% for the strongest baseline, within 2.4 points of oracle intent.

## Repository layout

```
BIND/
├── run_pilot.py, analyze.py, validate.py   # text-only pilot (supplement)
├── GEN_SPEC.md                             # pilot generation spec
└── embodied/
    ├── gen_episodes.py                     # AI2-THOR episode generator (B / C_rel / C_unr views, set-of-marks)
    ├── eps_json/                           # metadata of the 572 rendered episodes used in the paper
    ├── instr/
    │   ├── instructions.jsonl              # explicit split (fixed / dynamic / named)
    │   ├── grounded/instructions_grounded.jsonl        # memory-grounded split (REF / ATT / AMB)
    │   ├── para/instructions_grounded_notepara.jsonl   # paraphrased-note robustness set
    │   └── implicit/instructions_implicit.jsonl        # abandoned implicit-cue set (supplement)
    ├── run_vlm.py, run_vlm_x.py            # grounding: read-time draft + retry / re-plan on the current view
    ├── mode_classify.py                    # intent reasoner (REF vs ATT; --structured, --think, --ask, --n)
    ├── diag.py                             # exclusive failure taxonomy (Table 1), shared loader
    ├── analyze_vlm.py                      # accuracy per method / family / room, keyword baseline
    ├── route_analysis.py                   # routing policies (BIND, Freeze ID, Explicitize, oracle)
    ├── controls.py, nid_eval.py, note_cues.py, pm_eval.py, vn_eval.py   # baselines and controls
    ├── e2_summary.py                       # intent-module ablations (Table 3, Table S2)
    ├── e8_summary.py                       # clarification trade-off (Table S3, Fig. S3)
    ├── fam.py, export_rates.py, review_stats.py   # per-family results, CSV export, bootstrap CIs
    ├── build_note_variants.py, ingest_grounded.py # note ablation sets, ingestion of generated notes
    ├── train_intent.py                     # distilled cross-encoder intent classifier (optional)
    └── DESIGN.md, GROUNDED_SPEC.md, INSTR_SPEC.md # design and note-authoring specifications
```

## Environment

- Python ≥ 3.10; `ai2thor` 5.0 (iTHOR FloorPlans, `CloudRendering` headless on a GPU), `numpy`, `pillow`,
  `torch`, `transformers`, `vllm`.
- One 24 GB GPU runs the 8–9B models; Qwen3.6-27B and Gemma-4-31B need two 80 GB GPUs (`--tp 2`).

```bash
pip install "ai2thor>=5.0,<5.1" numpy pillow torch transformers vllm
```

## Reproducing the main results

```bash
cd embodied

# 1. Render the episodes (images are not stored in the repository)
python gen_episodes.py --scenes FloorPlan1 FloorPlan2 ... --seeds 0 --out episodes

# 2. Grounding with retry for one VLM on the memory-grounded split
python run_vlm.py --model <hf_id_of_Qwen3-VL-8B> --tag qwen3vl8b --episodes episodes \
    --instr instr/grounded/instructions_grounded.jsonl --backend vllm --out res_grounded

# 3. Intent reasoner (structured procedure, no thinking) = the "Ours" router
python mode_classify.py --model <hf_id_of_Qwen3.5-9B> --tag qwen3.5-9b --structured \
    --instr instr/grounded/instructions_grounded.jsonl --phrases eps_json --out modes

# 4. Failure taxonomy and accuracy (Table 1, Table 2)
python diag.py --files res_grounded/both/*.jsonl --instr instr/grounded/instructions_grounded.jsonl \
    --intent modes/qwen3.5-9b_grounded_struct.jsonl --label grounded --json diag_grounded.json

# 5. Ablations and clarification (Table 3, Tables S2–S3)
python e2_summary.py --modes modes_e2/*.jsonl --files res_grounded/both/*.jsonl \
    --instr instr/grounded/instructions_grounded.jsonl --out e2.csv
python e8_summary.py --policy never=... prompt_ask=... --sc sc_noask=... \
    --files res_grounded/both/*.jsonl --instr instr/grounded/instructions_grounded.jsonl --out e8.csv
```

| Paper item | Script |
|---|---|
| Table 1 (failure taxonomy), Fig. S5 | `diag.py` |
| Table 2 (cross-model accuracy) | `diag.py`, `analyze_vlm.py`, `export_rates.py` |
| Table 3, Table S2 (intent ablations, thinking) | `mode_classify.py --variant/--think`, `e2_summary.py` |
| Table S1 (robustness, 201-episode subset) | `run_vlm.py --temperature/--prompt-var`, `vn_eval.py` |
| Table S3, Fig. S3 (clarification) | `mode_classify.py --ask --n 5`, `e8_summary.py` |
| Fig. S4 (per family) | `fam.py` |

Decoding is greedy (temperature 0) with seed 2026 unless stated otherwise.

## External data and models (not included)

- **Simulator:** AI2-THOR 5.0 (`allenai/ai2thor`, PyPI `ai2thor`). Rendered images are regenerated with
  `gen_episodes.py`.
- **Models** (official Hugging Face releases): Qwen3-VL-8B, InternVL3.5-8B, Qwen3.6-27B, Gemma-4-31B (grounding);
  Qwen3.5-9B, Qwen3-8B, Qwen3-4B, Qwen3.5-4B, Llama-3.1-8B (intent reasoning). Weight checksums were not recorded at
  download time; check the provider's published checksums for the revision you use.
- Not included: model outputs, the labelled-answer key, and other cached results.

The instructions and memory notes were authored with an LLM following `INSTR_SPEC.md` and `GROUNDED_SPEC.md`;
labels come from simulator metadata and the construction rules.

---

## 中文说明

[English](#bind--embodieddrift) | **中文**

本仓库是论文 *EmbodiedDrift: When Retried Memory Writes Re-bind Visual References in Multi-Robot Shared Object
Memory* 的代码。

多个机器人共用一份物体记忆。某个机器人的写入因为别的机器人先提交而被拒绝时，常规做法是拿同一条指令对最新的记忆重试，
但这样可能悄悄改变指令指的是**哪一个物体**。**EmbodiedDrift** 是在 AI2-THOR 中构建的基准，用来测量这种由重试引起的
指代漂移；**BIND** 是按意图恢复的方法：指称性（referential）请求沿用读取时的物体身份，属性性（attributive）请求在当前
视图上重新定位，判断不了的就向用户提问。

主要结果（memory-grounded 划分，各条件下指令措辞完全相同，五个视觉定位模型取平均）：BIND 准确率 80–84%，最强基线
59–67%，与 oracle 意图只差 2.4 个百分点。

## 目录结构

- `run_pilot.py`、`analyze.py`、`validate.py`：纯文本试点实验（补充材料）。
- `embodied/gen_episodes.py`：AI2-THOR 场景生成（读取前 B、相关变化 C_rel、无关变化 C_unr 三种视图，带编号标记）。
- `embodied/eps_json/`：论文所用 572 个场景的元数据（物体、编号、状态、变化事件），不含渲染图像。
- `embodied/instr/`：指令与记忆笔记集，包括显式划分、memory-grounded 划分（REF / ATT / AMB）、笔记改写的稳健性集，
  以及补充材料中弃用的隐式线索集。
- `embodied/run_vlm.py`、`run_vlm_x.py`：视觉定位（读取时草稿 + 在当前视图上重试 / 重新规划）。
- `embodied/mode_classify.py`：意图推理器（区分 REF 与 ATT；`--structured`、`--think`、`--ask`、`--n` 等选项）。
- `embodied/diag.py`：排他式错误归因（表 1），也是各分析脚本共用的数据读取模块。
- `embodied/analyze_vlm.py`、`route_analysis.py`、`controls.py`、`nid_eval.py`、`note_cues.py`、`pm_eval.py`、
  `vn_eval.py`：各方法准确率、路由策略、基线与对照实验。
- `embodied/e2_summary.py`：意图模块消融（表 3、表 S2）；`embodied/e8_summary.py`：澄清权衡（表 S3、图 S3）。
- `embodied/fam.py`、`export_rates.py`、`review_stats.py`：按描述类别统计、导出 CSV、bootstrap 置信区间。
- `embodied/DESIGN.md`、`GROUNDED_SPEC.md`、`INSTR_SPEC.md`：设计与笔记撰写规范。

## 环境

- Python ≥ 3.10；`ai2thor` 5.0（iTHOR FloorPlans，GPU 上用 `CloudRendering` 无界面渲染）、`numpy`、`pillow`、
  `torch`、`transformers`、`vllm`。
- 8–9B 模型用一张 24 GB 显卡即可；Qwen3.6-27B 和 Gemma-4-31B 需要两张 80 GB 显卡（`--tp 2`）。

## 复现主结果

命令见上方英文部分「Reproducing the main results」，步骤依次为：

1. 用 `gen_episodes.py` 渲染场景图像（仓库不含图像）；
2. 用 `run_vlm.py` 对每个视觉模型在 memory-grounded 划分上做定位与重试；
3. 用 `mode_classify.py --structured` 运行意图推理器（即论文中的 Ours 路由）；
4. 用 `diag.py` 得到错误归因和准确率（表 1、表 2）；
5. 用 `e2_summary.py`、`e8_summary.py` 得到消融与澄清结果（表 3、表 S2–S3）。

论文图表与脚本的对应关系见英文部分的表格。除非另有说明，解码均为贪心（温度 0），随机种子 2026。

## 外部数据与模型（不在仓库内）

- **模拟器：** AI2-THOR 5.0（`allenai/ai2thor`，PyPI 包 `ai2thor`），场景图像用 `gen_episodes.py` 重新生成。
- **模型**（Hugging Face 官方发布版本）：视觉定位用 Qwen3-VL-8B、InternVL3.5-8B、Qwen3.6-27B、Gemma-4-31B；意图推理用
  Qwen3.5-9B、Qwen3-8B、Qwen3-4B、Qwen3.5-4B、Llama-3.1-8B。下载权重时没有记录校验值，请以所用版本在提供方页面公布的
  校验值为准。
- 不包含：模型输出、标注答案键及其他缓存结果。

指令和记忆笔记由 LLM 按 `INSTR_SPEC.md`、`GROUNDED_SPEC.md` 撰写；标签来自模拟器元数据和构造规则。
