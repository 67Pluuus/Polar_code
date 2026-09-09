# PoLar 论文第三方复现与实验扩展

> [!IMPORTANT]
> 本仓库是对论文 **《Skip a Layer or Loop It? Learning Program-of-Layers in LLMs》** 的第三方复现，不是论文作者维护的官方仓库，也不代表作者立场。原论文、原始代码和本仓库补充实现的归属与证据边界如下所列。

- 论文：[arXiv:2606.06574](https://arxiv.org/abs/2606.06574)
- 作者公开仓库：[tianyi-lab/PoLar](https://github.com/tianyi-lab/PoLar)
- 前期工作：[arXiv:2507.07996](https://arxiv.org/abs/2507.07996)

本仓库保留作者公开的 PoLar predictor、动态层执行和数学判分代码，并根据论文正文、附录和公开接口补充了 Stage One：在 DART-Math 上使用 MCTS 搜索 layer execution programs，生成 predictor 所需的 `merged_mcts_samples.json`。

**作者公开仓库没有提供 Stage One MCTS 源码，论文也没有给出复现所需的全部搜索超参数；本仓库无法声称使用作者原始配置，也不承诺精确复现论文数值。** 所有论文未明确给出的搜索参数均作为本项目可调整参数保存到运行记录中。

![PoLar 将预训练 Transformer 层组成可跳过或重复执行的程序。图片来自上游公开仓库。](search_space.png)

## 代码来源与责任边界

| 内容 | 来源 | 本仓库中的定位 |
|---|---|---|
| `polar/` | 作者公开仓库 | PoLar predictor、训练损失和 beam decoding；本复现按现有接口使用 |
| `llm_depth_router/` | 作者公开仓库 | 让基础 LLM 按指定层路径执行 |
| `dart_math/` | 作者公开仓库 | DART-Math 数据接口、答案抽取和数学等价判断 |
| `run_polar.py` | 作者公开仓库 | 原始 predictor 训练/评测入口，保留用于对照 |
| `stage_one/`、`run_stage_one.py` | 本仓库补充 | 数据准备、MCTS、分片、恢复、合并和验证 |
| `train_stage_one_predictor.py` | 本仓库补充 | 将显式数据划分接入现有 `polar/train.py`，并限制输出位置 |
| `run_robust_program_analysis.sh`、`run_12_paths_five_difficulties.sh` | 本仓库探索性扩展 | 固定路径的跨难度评测，不是论文作者公开实验脚本 |
| `representation_analysis.py` 及相关脚本 | 本仓库探索性扩展 | 借用 mNN/CKA 分析 residual geometry；不属于 PoLar 论文复现结论 |

本仓库没有修改基础 LLM 权重。MCTS、predictor 和固定路径评测都通过现有 `llm_depth_router` 改变 Transformer block 的执行顺序。

## 当前可以复现什么

主流程包括：

1. 从本地 DART-Math parquet 文件读取并按题去重；
2. 使用完整层路径执行 baseline；
3. 对 train/validation 题逐题运行 MCTS；
4. 对每条候选路径真实执行基础 LLM；
5. 使用现有 `dart_math` 逻辑抽取答案并判断数学等价性；
6. 保存每题的有效路径、无效路径、baseline 分数和搜索统计；
7. 合并并验证 predictor 可读取的 `merged_mcts_samples.json`；
8. 使用现有 `polar/data.py`、`polar/model.py` 和 `polar/train.py` 训练 predictor。

以下内容不能从本仓库直接推出：

- MCTS 找到的是全局最优路径；
- 每条搜索到的正确路径都能迁移到其他题目或模型；
- 本项目默认超参数等同于作者实验配置；
- 小规模 smoke test 能复现论文准确率；
- residual 相似性能够证明 skip/loop 导致正确率变化；
- 同一模型不同执行路径已经收敛到“柏拉图表征”。

## 目录约定

所有命令在同时包含 `./Polar_code` 和 `./Polar_data` 的项目根目录执行：

```text
./
├── Polar_code/                 # 本仓库
└── Polar_data/                 # 模型、数据、缓存和全部运行产物
    ├── models/
    ├── raw/
    ├── cache/
    ├── runtime/
    └── runs/
```

本复现流程的所有产物必须位于 `./Polar_data`。不要把正式运行产物写入 `./Polar_code/outputs`。

主要代码结构：

```text
Polar_code/
├── polar/                      # 作者公开的 predictor 实现
├── llm_depth_router/           # 作者公开的动态层执行实现
├── dart_math/                  # 作者公开的数学答案判定实现
├── stage_one/                  # 本仓库补充的 MCTS Stage One
├── run_stage_one.py            # Stage One 分阶段入口
├── run_stage_one_pipeline.sh   # Stage One 串联脚本
├── train_stage_one_predictor.py
└── check_stage_one_static.py
```

## 用户需要准备的内容

本仓库不会自动下载模型、数据集、checkpoint 或依赖。以 Llama-3.2-3B-Instruct 复现为例，需要准备：

```text
./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct/
./Polar_data/raw/dart-math-pool-math/data/
./Polar_data/cache/huggingface/hub/
```

对应资源：

- 基础模型：`meta-llama/Llama-3.2-3B-Instruct`；
- 数据集：`hkust-nlp/dart-math-pool-math` 的五个 parquet 分片；
- predictor embedding model：`Qwen/Qwen3-Embedding-0.6B`；
- Python 依赖：`./Polar_code/stage_one/requirements.txt`。

模型和数据各自受其原始许可证及访问条件约束。

## 安装依赖

在 Linux 环境中运行：

```bash
python -m pip install -r ./Polar_code/stage_one/requirements.txt
```

这里使用 `-r`。`requirements.txt` 不是可编辑 Python 包，不能写成 `pip install -e stage_one/requirements.txt`。

## 先做静态检查

该检查不加载模型，也不运行推理：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B ./Polar_code/check_stage_one_static.py \
  --run-name static_review \
  --clean
```

输出位于 `./Polar_data/runs/static_review/environment/static_report.json`。

## 小规模流程检查

下面的命令仅检查数据、MCTS、合并、验证和恢复机制能否跑通，不用于报告论文结果：

```bash
CUDA_VISIBLE_DEVICES=1 bash ./Polar_code/run_stage_one_pipeline.sh \
  --nproc_per_node=1 \
  --run-name mcts_smoke \
  --data-path ./Polar_data/raw/dart-math-pool-math \
  --source-revision local-files \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --model-path ./Polar_data/models/meta-llama/Llama-3.2-3B-Instruct \
  --model-revision local-snapshot \
  --max-questions-per-diff 8 \
  --difficulties "1" \
  --train-predictor false \
  --predictor-config ./Polar_code/stage_one/predictor_config.json \
  --clean
```

输出位于 `./Polar_data/runs/mcts_smoke/`。首次运行可使用 `--clean`；中断恢复时删除 `--clean`，程序会验证并跳过已经完成的逐题结果。

## 正式运行

正式的单卡、八卡、独立合并、数据验证和 predictor 训练命令见：

- [Stage One 使用说明](./STAGE_ONE_使用说明.md)
- [跨难度固定路径分析说明](./ROBUST_SKIP_LOOP_分析说明.md)
- [12 条固定路径五难度实验](./FIXED_12_PATHS_五难度说明.md)
- [Residual 表征可视化说明](./REPRESENTATION_可视化说明.md)

第一阶段采用一进程一卡的数据并行方式，不使用 DDP，也不进行梯度同步。每个 rank 处理确定且互斥的题目分片，并独立保存逐题结果。

## MCTS 监督与 Predictor 的关系

MCTS 对每道题搜索多个 layer execution programs。只有真实执行基础 LLM 后得到正确答案的路径才进入：

```text
final_valid_transitions
```

`polar/data.py` 会将同一道题的多条有效路径分别展开为训练样本。Predictor 学习的是问题条件下的 segmentation 和 `skip/keep/repeat` 操作概率，而不是一条经过证明的全局最优路径。

推理时 beam search 可以组合出训练集中没有出现过的新路径。结构合法不代表答案正确，因此未知路径仍需要执行基础 LLM 才能得到真实 reward。

## `merged_mcts_samples.json`

每个难度最终生成一个文件：

```text
./Polar_data/runs/<run-name>/merged/<model-id>/dart-math-diff-<1..5>/merged_mcts_samples.json
```

核心格式：

```json
{
  "samples": [
    {
      "sample_id": "...",
      "question": "Solve ...",
      "gt_ans": "\\boxed{42}",
      "initial_score": 1,
      "final_valid_transitions": [
        [0, 1, 2, 4, 5, 6],
        [0, 1, 2, 2, 3, 4, 5]
      ],
      "final_invalid_transitions": [
        [0, 1, 3, 4, 5]
      ]
    }
  ]
}
```

- 跳过某层：路径中不出现该层索引；
- 重复某层或连续层段：相应索引重复出现；
- `initial_score`：完整层 baseline 的二值正确性；
- `final_valid_transitions`：实际执行且判定正确的路径；
- `final_invalid_transitions`：实际执行且判定错误的路径。

标准 predictor 训练读取有效路径。无效路径主要用于数据审计和评测缓存，并没有直接作为标准 Predictor 损失中的负路径样本。

## 结果应如何表述

建议使用以下表述：

> 本实验基于作者公开的 PoLar predictor 与动态层执行代码，按照论文描述独立补充 MCTS 数据构造流程。搜索超参数为本项目配置，实验结果属于第三方复现结果。

不要使用以下表述：

> 本仓库是 PoLar 官方实现；使用了作者原始 MCTS 配置；已经精确复现论文结果。

## 引用与致谢

使用本仓库时，请优先引用原论文并注明作者公开仓库。本仓库的 Stage One 和分析脚本只是第三方复现工作，不能替代对原论文的引用。

```bibtex
@inproceedings{li2026PoLar,
  author    = {Ziyue Li and Yang Li and Tianyi Zhou},
  title     = {Skip a Layer or Loop It? Learning Program-of-Layers in LLMs},
  booktitle = {International Conference on Machine Learning},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.06574}
}
```

前期工作：

```bibtex
@article{li2025CoLa,
  title   = {Skip a Layer or Loop It? Test-Time Depth Adaptation of Pretrained LLMs},
  author  = {Ziyue Li and Yang Li and Tianyi Zhou},
  journal = {arXiv preprint arXiv:2507.07996},
  year    = {2025}
}
```
