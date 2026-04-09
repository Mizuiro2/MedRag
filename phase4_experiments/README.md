# Phase4 实验

按论文 Table 2 进行 HyKGE vs KGRAG 对比实验（EM、PCR）。

## 数据集

- **MMCU-Medical**: `datasets/MMCU-Medical.xlsx`（15 个 sheet：医学三基、药理学、护理学等，共 2819 题）
- **CMB-Exam**: `datasets/CMB-Exam.json` + `CMB-Exam-answers.json`（随机抽样 4000 题）

## 方法

- **HyKGE**: 复用 phase2 完整流程（HO → NER → Entity Linking → KG Retrieval → Rerank → LLM）
- **KGRAG**: 基线，仅用 query 检索，无 HO、无 rerank

## 指标（论文 Appendix A.2.2）

- **EM**: Exact Match，预测与标准答案完全一致
- **PCR**: Partial Correct Rate，仅针对多选题；若无错误答案（预测是标准答案的子集）则判为正确

## 运行

```bash
cd h:\MedRAG\phase4_experiments
G:\Anaconda\envs\rag_fyp\python.exe run_experiments.py --model ds
```

### DeepSeek 需代理时

若 DeepSeek 出现 Connection error，可用代理脚本运行（默认端口 7897，可在脚本内修改）：

```bash
# 方式一：bat
run_with_proxy.bat -m ds

# 方式二：PowerShell
.\run_with_proxy.ps1 -m ds
```

### 参数

- **`--model` / `-m`**（必选）: 选择 LLM
  - `ds` - DeepSeek
  - `qwen` - Qwen3-max
  - `doubao` - Doubao
- `--limit N`: 每个数据集最多测试 N 条（调试用）
- `--cmb-exam-size N`: CMB-Exam 抽样数量（默认 4000）
- `--output` / `-o`: 结果输出路径（默认 `table2_results.txt`）

### 示例

```bash
# 使用 DeepSeek 完整实验
G:\Anaconda\envs\rag_fyp\python.exe run_experiments.py -m ds

# 使用 Qwen3-max
G:\Anaconda\envs\rag_fyp\python.exe run_experiments.py -m qwen

# 快速测试（每数据集 5 题）
G:\Anaconda\envs\rag_fyp\python.exe run_experiments.py -m ds --limit 5
```

## 依赖

需先安装 phase2 依赖，并额外安装：

```bash
G:\Anaconda\envs\rag_fyp\python.exe -m pip install openpyxl openai
```

## 输出

结果写入 `table2_results.txt`，包含 HyKGE vs KGRAG 在 MMCU-Medical、CMB-Exam 上的 EM、PCR。
