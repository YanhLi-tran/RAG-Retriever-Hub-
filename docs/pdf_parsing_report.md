# PDF 解析实验报告

## 1. 实验目标

将 RAG_Data/papers/ 中的 7 篇 AI 论文 PDF 解析为干净的 Markdown 文本，用于后续 RAG 知识库构建。

**核心要求：**
- 表格、公式等结构化内容需保留
- 输出格式为 Markdown，便于后续分块和向量化
- 离线处理，可用时间换精度

## 2. 数据源

| 文件 | 页数 | 大小 | 内容特点 |
|------|------|------|----------|
| bert.pdf | 16 | 757 KB | 经典论文，含表格 |
| gpt4.pdf | 100 | 5.1 MB | 技术报告，大量表格和代码 |
| llama2.pdf | 77 | 13.6 MB | 大型论文，表格密集 |
| lora.pdf | 26 | 1.6 MB | 方法论文 |
| rag_original.pdf | 19 | 885 KB | RAG 原始论文 |
| ragas.pdf | 8 | 232 KB | 评估方法论文 |
| transformer.pdf | 15 | 2.2 MB | 经典论文 |

## 3. 方案探索过程

### 3.1 方案一：PyMuPDF（放弃）

**结果：**
- 文本提取完整，字符数合理
- 但表格结构丢失（行列对齐关系丢失）
- 公式提取不完整

**结论：** 作为基础文本提取可用，但无法满足表格和公式需求。

### 3.2 方案二：Unstructured hi_res（放弃）

**尝试过程：**
1. 安装 `unstructured[pdf]`
2. 使用 `fast` 策略测试 → 表格未被识别，开头有乱码
3. 改用 `hi_res` 策略 → 需要 Tesseract OCR

**遇到的问题：**

| 问题 | 描述 | 解决尝试 |
|------|------|----------|
| Tesseract 未安装 | hi_res 策略依赖 Tesseract OCR 引擎 | 官方下载 403，conda 安装超时 |
| 安装 rapidocr | 尝试用 rapidocr 替代 tesseract | unstructured 内部仍调用 tesseract |
| HuggingFace 模型下载 | hi_res 需要下载布局检测模型 | 网络问题无法下载 |

**结论：** 依赖系统级程序 Tesseract，安装困难，放弃。

### 3.3 方案三：pymupdf4llm（最终采用）

**优势：**
- 基于 PyMuPDF 的 LLM 专用版本
- 自带 RapidOCR（纯 Python，无需系统依赖）
- 直接输出 Markdown 格式
- 表格保留 `|` 结构，公式保留原始写法

**核心代码：**

```python
import pymupdf4llm

# 单文件解析
md_text = pymupdf4llm.to_markdown(pdf_path)

# 保存为 Markdown 文件
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_text)
```

**完整脚本逻辑：**

```python
def export_pdf_to_markdown(pdf_path: str, output_path: str):
    """解析单个 PDF 并导出为 Markdown 文件"""
    # to_markdown 会自动调用 RapidOCR 处理扫描页
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"<!-- PDF文件: {os.path.basename(pdf_path)} -->\n")
        f.write(f"<!-- 总字符数: {len(md_text)} -->\n\n")
        f.write(md_text)
```

## 4. 输出质量分析

### 4.1 表格解析效果对比

**PyMuPDF 原生提取（无表格结构）：**
```
Task
Train
Development
Test
Natural Questions
79169
8758
3611
TriviaQA
78786
8838
11314
```

**pymupdf4llm 提取（保留 Markdown 表格结构）：**
```
|Exam|GPT-4|GPT-4 (no vision)|GPT-3.5|
|---|---|---|---|
|Uniform Bar Exam (MBE+MEE+MPT)|298 / 400 (~90th)|298 / 400 (~90th)|213 / 400 (~10th)|
|SAT Evidence-Based Reading & Writing|710 / 800 (~93rd)|710 / 800 (~93rd)|670 / 800 (~87th)|
|SAT Math|700 / 800 (~89th)|690 / 800 (~89th)|590 / 800 (~70th)|
```

**对比总结：**

| 方案 | 表格结构 | 列对齐 | 可直接用于 RAG |
|------|----------|--------|----------------|
| PyMuPDF 原生 | ❌ 丢失 | ❌ | ❌ 需额外处理 |
| pymupdf4llm | ✅ 保留 | ✅ | ✅ 可直接使用 |

### 4.3 原始输出问题

pymupdf4llm 输出的 Markdown 存在以下噪声：

| 问题 | 严重程度 | 示例 | 影响 |
|------|----------|------|------|
| 图表OCR文字 | ★★★ | `<!-- Start of picture text -->` 标记的内容 | 混入正文，污染检索 |
| 脚注链接 | ★★ | 参考文献列表、URL | 无关信息 |
| 多余空行 | ★ | 连续多个空行 | 浪费 token，影响分块 |

### 4.4 清理方案

**第一轮清理：** md_cleaner.py

| 清理项 | 处理方式 |
|--------|----------|
| 图表说明行 | 跳过 `Figure X:` / `Table X:` 开头的独立行 |
| 参考文献区域 | 检测到 `References` 标题后跳过整个区域 |
| 脚注链接 | 跳过纯 URL 行、`[数字]` 格式的引用条目 |
| 行内脚注标记 | 移除 `[数字]` 格式标记 |
| 多余空行 | 连续空行合并为 1 个 |

**清理效果（以 rag_original.md 为例）：**

| 指标 | 清理前 | 清理后 | 减少 |
|------|--------|--------|------|
| 行数 | 489 | 303 | 38% |
| 字符数 | 72,982 | 47,166 | 35% |

**第二轮清理：** 手动清理 `<!-- Start of picture text -->` 标记的图片 OCR 文字

这些内容是论文中架构图、流程图里提取出来的文字，混入正文会污染检索结果。


### 4.5 特殊问题：双栏表格解析失败

**问题描述：**

部分学术论文（如 `rag_original.pdf`）采用双栏排版，表格横跨两栏或并排两个表格。pymupdf4llm 按坐标读取文字时，会将左右两栏的内容混在一起，导致表格结构完全错乱。

**问题表现（rag_original.md Table 1 & Table 2）：**

```markdown
# 错误输出：左右两栏数据混在一起
||Model|NQ|TQA|WQ|CT||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Cld|T511B 52|345|/501|374||Model|Jeop|ardy|MSM|ARCO|FVR3|FVR2|
|ose
Book|- []
T5-11B+SSM|.
36.6|- .
- /60.5|.
44.7|-
-||B-1|QB-1|R-L|B-1|Label|Acc.|
|Open|REALM |40.4|- / -|40.7|46.8|SotA|-|-|**49.8* **|**49.9***|**76.8**|**92.2***|
```

**原因分析：**

```
原文 PDF 布局：
┌─────────────────────┬─────────────────────┐
│ Table 1: QA Scores  │ Table 2: Generation │
│ Model | NQ | TQA    │ Model | Jeopardy    │
│ T5-11B | 34.5 | -   │ BART  | 15.1       │
│ DPR    | 41.5 | 57.9│ RAG-T | 17.3       │
└─────────────────────┴─────────────────────┘

pymupdf4llm 按坐标读取：
→ 左栏第一行 + 右栏第一行 → 拼成一行
→ 数据完全混乱，无法还原
```

**尝试过的自动修复方案（均失败）：**

| 方案 | 方法 | 结果 |
|------|------|------|
| pdfplumber | `page.extract_tables()` | 双栏表格提取为空或碎片 |
| Camelot lattice | 表格线检测 | 无网格线的表格无法识别 |
| Camelot stream | 文字对齐检测 | 双栏文字对齐混乱 |
| 调整 pymupdf4llm 参数 | 修改 `write_images` 等选项 | 无法解决双栏问题 |

**最终方案：手动重建表格**

由于只有 `rag_original.pdf` 存在此问题（1 个文件），采用手动修复：

```python
# 从原始 PDF 中人工读取表格内容，重建为正确的 Markdown 表格
old_broken_table = """||Model|NQ|TQA|WQ|CT||||||||
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Cld|T511B 52|345|/501|374||Model|Jeop|ardy|..."""

new_fixed_table = """**Table 1: Open-Domain QA Test Scores**

| Model | NQ | TQA | WQ | CT |
|---|---|---|---|---|
| Closed T5-11B | 34.5 | - / 50.1 | 37.4 | - |
| Book T5-11B+SSM | 36.6 | - / 60.5 | 44.7 | - |
| Open REALM | 40.4 | - / - | 40.7 | 46.8 |
| Book DPR | 41.5 | **57.9** / - | 41.1 | 50.6 |
| RAG-Token | 44.1 | 55.2 / 66.1 | **45.5** | 50.0 |
| RAG-Seq. | **44.5** | 56.8 / **68.0** | 45.2 | **52.2** |"""

content = content.replace(old_broken_table, new_fixed_table)
```

**修复范围：**

| 表格 | 问题 | 修复方式 |
|------|------|----------|
| Table 1 (QA Scores) | 双栏混排 | 手动重建为左右分离的两个表格 |
| Table 2 (Generation) | 双栏混排 | 手动重建为左右分离的两个表格 |
| Table 3 (生成示例) | 列错位 | 手动重建，按 Task/Input/Model/Generation 排列 |
| Table 4 & 5 (评估) | 数据错位 | 手动重建为独立表格 |

**修复效果：**

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 坏表格数 | 4 | 0 |
| 列数一致性 | 不一致 | 一致 |
| 数据可读性 | ❌ 无法理解 | ✅ 完整可读 |

**经验总结：**

1. **双栏表格是 PDF 解析的已知难题**：目前没有完美的自动化方案
2. **影响范围有限**：7 篇论文中只有 1 篇（rag_original.pdf）存在此问题
3. **手动修复成本可控**：只有 4 个表格需要重建，耗时约 15 分钟
4. **后续预防**：遇到双栏论文时，优先检查表格是否被正确解析

## 5. 最终输出

### 5.1 技术选型总结

| 维度 | PyMuPDF | Unstructured hi_res | pymupdf4llm |
|------|---------|---------------------|-------------|
| 安装难度 | ⭐ 简单 | ⭐⭐⭐ 复杂 | ⭐ 简单 |
| 系统依赖 | 无 | Tesseract OCR | 无 |
| 表格支持 | ❌ 丢失结构 | ✅ 支持 | ✅ 保留结构 |
| 公式支持 | ⚠️ 部分 | ✅ 支持 | ⚠️ 部分 |
| OCR 能力 | ❌ 无 | ✅ 内置 | ✅ 内置 RapidOCR |
| 输出格式 | 纯文本 | 结构化元素 | Markdown |
| 解析速度 | ⭐⭐⭐ 快 | ⭐ 慢 | ⭐⭐ 中 |
| 离线可用 | ✅ | ⚠️ 需下载模型 | ✅ |
| **最终选择** | ❌ | ❌ | ✅ |

**选择 pymupdf4llm 的核心原因：**
1. 零系统依赖，纯 Python 方案
2. 表格结构完整保留，可直接用于 RAG
3. 自带 OCR，离线环境可用
4. 输出 Markdown，便于后续处理

### 5.3 清理效果汇总

| 文件 | 原始字符 | 清理后字符 | 减少比例 |
|------|----------|------------|----------|
| bert.md | 66,263 | 62,611 | 5.5% |
| gpt4.md | 299,909 | 255,417 | 14.8% |
| llama2.md | 286,036 | 285,541 | 0.2% |
| lora.md | 90,080 | 69,517 | 22.8% |
| rag_original.md | 72,982 | 47,166 | 35.4% |
| ragas.md | 32,911 | 32,409 | 1.5% |
| transformer.md | 42,449 | 32,916 | 22.5% |

## 6. 技术选型决策总结

### 6.1 决策流程

```
需求分析：PDF → 干净 Markdown，保留表格结构
    ↓
方案调研：PyMuPDF / Unstructured / pymupdf4llm
    ↓
快速验证：各方案小规模测试
    ↓
问题发现：PyMuPDF 表格丢失、Unstructured 依赖复杂
    ↓
最终选择：pymupdf4llm（零依赖 + 表格保留 + 内置 OCR）
    ↓
质量优化：两轮清理（自动 + 手动）
```

### 6.2 关键决策点

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 文本提取库 | pymupdf4llm | 表格结构保留最好 |
| OCR 方案 | 内置 RapidOCR | 避免 Tesseract 系统依赖 |
| 输出格式 | Markdown | 便于后续分块和向量化 |
| 清理策略 | 分步处理 | 先输出再清理，便于调试 |

### 6.3 经验教训

1. **先验证核心需求**：表格保留是核心需求，PyMuPDF 原生无法满足
2. **警惕系统依赖**：Unstructured hi_res 依赖 Tesseract，在 Windows 上安装困难
3. **分步处理更可控**：先输出原始结果，再单独清理，便于定位问题
4. **手动清理不可避免**：OCR 提取的图片文字需要手动规则处理

## 7. 经验总结

1. **pymupdf4llm 是最佳选择**：纯 Python 方案，自带 OCR，输出 Markdown，适合离线 RAG 场景

2. **unstructured hi_res 依赖过重**：需要 Tesseract 系统程序 + HuggingFace 模型下载，安装复杂

3. **清理是必要的**：OCR 提取的文字包含大量噪声（图表说明、脚注、图片文字），直接使用会污染检索

4. **分步处理便于调试**：先生成原始输出，再单独清理，便于定位问题和调整清理规则

5. **llama2 清理效果低的原因**：
   - 图表说明行仅 3 行，参考文献行 0 行（与其他论文相比极少）
   - 表格内容占比 13.7%，属于有用数据，不应清理
   - 主要内容为正文段落，噪声本身就少
   - 说明该论文格式规范，OCR 提取质量高

## 8. 后续优化方向

1. **表格结构化**：目前表格转为 Markdown 文本，可考虑后续用 LLM 进一步结构化
2. **公式识别**：数学公式的 Markdown 表示可进一步标准化（如转为 LaTeX）
3. **图片描述**：可结合多模态模型为图片生成描述文字，补充纯文本提取的不足
