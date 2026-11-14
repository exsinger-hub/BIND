<div align="center">

# BIND

**面向共享物体记忆的多机器人：按意图决定如何重试**

[English](README.md) | **简体中文**

</div>

多个机器人读写同一份物体记忆时，一个机器人的写入可能因为别的机器人先提交而被拒绝。常规做法是拿同一条指令对最新的
记忆重试，但这期间场景可能已经变了，重试会悄悄选中一个和用户本意不同的物体。

BIND 在重试之前先判断用户到底指的是什么：

- **指称性（referential）**请求，用户指的是*那一个具体物体*：沿用读取时选中的物体；
- **属性性（attributive）**请求，用户指的是*符合描述的那个物体*：在当前视图上重新定位；
- **判断不了**：向用户提问。

仓库里还包含 **EmbodiedDrift**：一个在 AI2-THOR 中构建、用来测量这种重试导致的指代漂移的基准。

<p align="center"><img src="assets/teaser.png" width="460" alt="同一场景下 Retry、Freeze ID 与 BIND 的对比"></p>

## 方法

<p align="center"><img src="assets/overview.png" width="900" alt="BIND 流程"></p>

1. 读取时，视觉定位模型在视图 *B* 上生成动作草稿，并记下选中的物体编号。
2. 提交被拒绝后，一个只看文本的意图推理器读取对话和共享笔记（不看图像），把请求判为指称性、属性性或不确定。
3. 按判断结果重试：沿用读取时的编号、在当前视图 *C* 上重新定位，或者向用户提问。

## 结果

<p align="center"><img src="assets/qualitative.png" width="900" alt="定性示例"></p>

<p align="center"><img src="assets/radar.png" width="430" alt="各方法表现"></p>

在基准的 memory-grounded 划分上，五个视觉定位模型取平均，BIND 的准确率约为 80–84%，最强的重试基线为 59–67%。
直接重试会做错指称性请求，一律沿用旧编号会做错属性性请求；BIND 两种都能处理。

## 安装

```bash
git clone https://github.com/exsinger-hub/BIND.git
cd BIND
pip install "ai2thor>=5.0,<5.1" numpy pillow torch transformers vllm
```

- Python ≥ 3.10。AI2-THOR 在 GPU 上用 `CloudRendering` 无界面渲染。
- 8–9B 模型一张 24 GB 显卡即可；27–31B 模型需要两张 80 GB 显卡（`--tp 2`）。

## 快速上手

```bash
cd embodied

# 1. 根据保存的元数据渲染场景（仓库不含图像）
python gen_episodes.py --scenes FloorPlan1 FloorPlan2 --seeds 0 --out episodes

# 2. 用视觉模型做定位与重试
python run_vlm.py --model <vlm_model_id> --tag myvlm --episodes episodes \
    --instr instr/grounded/instructions_grounded.jsonl --backend vllm --out res

# 3. 意图推理器（结构化提示）
python mode_classify.py --model <llm_model_id> --tag myllm --structured \
    --instr instr/grounded/instructions_grounded.jsonl --phrases eps_json --out modes

# 4. 准确率与错误分类
python diag.py --files res/both/*.jsonl --instr instr/grounded/instructions_grounded.jsonl \
    --intent modes/myllm_grounded_struct.jsonl --label grounded --json report.json
```

`mode_classify.py` 常用选项：`--think`（开启模型思考）、`--ask`（允许回答"不确定"）、
`--n 5 --temperature 0.7`（多次采样投票）、`--variant`（笔记消融）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `embodied/gen_episodes.py` | AI2-THOR 场景生成（同类重复物体、变化前后视图、编号标记） |
| `embodied/eps_json/` | 572 个场景的元数据 |
| `embodied/instr/` | 指令与记忆笔记：显式集、memory-grounded 集（REF / ATT / AMB）、改写笔记集、隐式线索集 |
| `embodied/run_vlm.py`、`run_vlm_x.py` | 定位与重试（HF 或 vLLM 后端） |
| `embodied/mode_classify.py` | 意图推理器 |
| `embodied/diag.py` | 排他式错误分类；公共数据读取模块 |
| `embodied/analyze_vlm.py`、`route_analysis.py`、`controls.py`、`nid_eval.py`、`note_cues.py`、`pm_eval.py`、`vn_eval.py` | 准确率、路由策略、基线与对照 |
| `embodied/e2_summary.py`、`e8_summary.py` | 意图模块消融；澄清提问的权衡 |
| `embodied/fam.py`、`export_rates.py`、`review_stats.py` | 按描述类别统计、导出 CSV、bootstrap 区间 |
| `embodied/build_note_variants.py`、`ingest_grounded.py`、`train_intent.py` | 笔记变体、数据导入、可选的蒸馏分类器 |
| `embodied/*_SPEC.md`、`DESIGN.md` | 基准设计与笔记撰写规范 |
| `run_pilot.py`、`analyze.py`、`validate.py` | 纯文本试点实验 |

## 数据与模型

- 模拟器：[AI2-THOR](https://github.com/allenai/ai2thor) 5.0，iTHOR FloorPlans；图像用 `gen_episodes.py` 重新生成。
- 实验中用到的视觉定位模型：Qwen3-VL-8B、InternVL3.5-8B、Qwen3.6-27B、Gemma-4-31B；意图推理器：Qwen3.5-9B、
  Qwen3-8B、Qwen3-4B、Qwen3.5-4B、Llama-3.1-8B。`--model` 可以传入 `transformers` 或 `vllm` 支持的任意 Hugging Face 模型。
- 指令和笔记由 LLM 按 `embodied/INSTR_SPEC.md`、`embodied/GROUNDED_SPEC.md` 撰写；目标物体和标签来自模拟器元数据。
- 不包含模型输出和缓存结果。
