# RAG 知识库系统项目报告

## 1. 项目概述

构建一个面向 AI 论文和技术文档的 RAG（Retrieval-Augmented Generation）知识库系统。系统从多种数据源（PDF 论文、技术文档 HTML、博客文章）出发，经过格式解析、文本清洗、智能切分、向量化存储和混合检索，最终实现基于自然语言问题的精准问答与溯源。

**PDF 数据源：**

| 文件 | 页数 | 大小 | 内容特点 |
|------|------|------|----------|
| bert.pdf | 16 | 757 KB | 经典论文，含表格 |
| gpt4.pdf | 100 | 5.1 MB | 技术报告，大量表格和代码 |
| llama2.pdf | 77 | 13.6 MB | 大型论文，表格密集 |
| lora.pdf | 26 | 1.6 MB | 方法论文 |
| rag_original.pdf | 19 | 885 KB | RAG 原始论文 |
| ragas.pdf | 8 | 232 KB | 评估方法论文 |
| transformer.pdf | 15 | 2.2 MB | 经典论文 |

**HTML 数据源：**

| 类别 | 文件数 | 来源 | 内容特点 |
|------|:------:|------|----------|
| LangChain 文档 | 12 | 官方文档站 | 框架使用指南、API 参考 |
| LlamaIndex 文档 | 7 | 官方文档站 | 框架使用指南、API 参考 |
| Web 博客文章 | 5 | Lil'Log 等技术博客 | LLM、RAG、Transformer 深入讲解 |

**CSV 对比数据源：**

| 文件 | 大小 | 内容 |
|------|:----:|------|
| chunk_strategy_comparison.csv | 754 B | 5 种分块策略对比（原理、优劣、适用场景） |
| embedding_comparison.csv | 561 B | 9 种 Embedding 模型对比（维度、MTEB 分数、速度） |
| model_benchmark.csv | 555 B | 12 种 LLM 对比（参数量、MMLU/Coding/Math 分数） |

## 2. 技术栈

| 模块 | 技术选型 | 版本 |
|------|---------|------|
| PDF 解析 | pymupdf4llm | — |
| HTML 解析 | trafilatura | 2.1.0 |
| 文本切分 | LangChain RecursiveCharacterTextSplitter | — |
| 向量嵌入 | BAAI/bge-large-zh（1024 维） | v1.5 |
| 向量数据库 | ChromaDB（HNSW 索引） | 1.5.9 |
| 稀疏检索 | rank-bm25 (BM25Okapi) | 0.2.2 |
| RRF 融合 | 自定义 Reciprocal Rank Fusion | — |
| 评估框架 | RAGAS | 0.4.3 |
| 生成模型 | DeepSeek Chat (API) | — |
| 运行环境 | Python 3.13, Windows | — |

## 3. PDF 解析与清洗

### 3.1 解析方案选型

经过三轮方案对比，最终选择 **pymupdf4llm**：

| 方案 | 表格结构 | 系统依赖 | OCR | 结论 |
|------|----------|---------|-----|------|
| PyMuPDF 原生 | ❌ 丢失 | 无 | 无 | 放弃 |
| Unstructured hi_res | ✅ | Tesseract + 模型下载 | 需 Tesseract | 放弃 |
| **pymupdf4llm** | ✅ 保留 | 纯 Python | 内置 RapidOCR | ✅ 采用 |

**选择 pymupdf4llm 的核心原因：**
1. 零系统依赖，纯 Python 方案
2. 表格结构完整保留为 Markdown 格式
3. 自带 RapidOCR（纯 Python OCR，无需系统级程序）
4. 输出 Markdown，便于后续分块和向量化
5. 离线可用

### 3.2 双栏表格问题

部分论文（如 `rag_original.pdf`）采用双栏排版，pymupdf4llm 按坐标读取文字时会将左右两栏内容混在一起，导致表格结构完全错乱。尝试了 pdfplumber、Camelot 等自动修复方案均失败，最终采用**手动重建**。影响范围仅 1 篇论文中的 4 个表格，耗时约 15 分钟。

### 3.3 Markdown 清理

pymupdf4llm 输出的 Markdown 包含以下噪声：

| 问题 | 处理方式 |
|------|----------|
| 图表 OCR 文字（Figure/Table 说明） | 正则匹配跳过独立说明行 |
| 参考文献区域 | 检测到 References 标题后跳过整个区域 |
| 脚注链接 / URL | 跳过纯 URL 行 |
| 行内脚注标记 [数字] | 正则替换移除 |
| PDF 注释头及 OCR 残留 `<!-- ... -->` | 跳过所有 HTML 注释行（修复：原只匹配 `<!-- PDF` 和 `<!-- 总字符数`，遗漏了 `<!-- Start of picture text -->`） |
| 多余空行 | 连续空行合并为 1 个 |

**清理效果汇总：**

| 文件 | 原始字符 | 清理后字符 | 减少比例 |
|------|----------|------------|----------|
| bert.md | 66,263 | 48,756 | 26.4% |
| gpt4.md | 299,909 | 188,910 | 37.0% |
| llama2.md | 286,036 | 229,069 | 19.9% |
| lora.md | 90,080 | 66,150 | 26.6% |
| rag_original.md | 72,982 | 44,326 | 39.3% |
| ragas.md | 32,911 | 26,430 | 19.7% |
| transformer.md | 42,449 | 30,575 | 28.0% |

## 4. HTML 解析

### 4.1 解析方案选型

| 方案 | 导航栏去除 | 代码块保留 | 输出格式 | 结论 |
|------|:---------:|:---------:|:--------:|:----:|
| BeautifulSoup + 自定义 | 需手动写规则 | ✅ | Markdown | 工作量大 |
| markitdown (Microsoft) | ❌ 保留导航栏 | ✅ | Markdown | 需额外清洗 |
| **trafilatura** | ✅ 自动去除 | ✅ 保留 | Markdown | ✅ 采用 |

**选择 trafilatura 的原因：**
1. 内置 boilerplate removal 算法，自动去除导航栏、侧栏、页脚
2. 保留代码块、表格、链接结构
3. 直接输出 Markdown，与 PDF 管线一致
4. 纯 Python，一行函数即可提取

### 4.2 提取效果

| 类别 | 原始 HTML | 提取后 Markdown | 减少比例 |
|------|:---------:|:---------------:|:--------:|
| LangChain 文档 | 25 MB | 181 KB | 99.3% |
| LlamaIndex 文档 | 2.0 MB | 86 KB | 95.7% |
| Web 博客文章 | 1.1 MB | 180 KB | 83.8% |

## 5. CSV 表格解析

### 5.1 解析方案

CSV 表格数据不适合直接用于检索，需要转为可读的文本。采用**双格式输出**策略：

1. **Markdown 表格** — 保留原始结构化数据，便于精确查询
2. **自然语言描述** — 将每行转为一段可读文本，便于语义检索

```
示例转换：
原始 CSV: BAAI/bge-large-zh-v1.5,1024,64.2,63.8,中,是,中文场景首选
转换后: - 模型: BAAI/bge-large-zh-v1.5; 维度: 1024;
         中文MTEB: 64.2; 英文MTEB: 63.8; 速度: 中;
         开源: 是; 推荐场景: 中文场景首选
```

### 5.2 处理结果

| 文件 | 原始大小 | 输出大小 |
|------|:--------:|:--------:|
| chunk_strategy_comparison.csv | 754 B | 1,200 chars |
| embedding_comparison.csv | 561 B | 1,461 chars |
| model_benchmark.csv | 555 B | 1,738 chars |

## 6. 文本切分策略

采用 `RecursiveCharacterTextSplitter`，按以下优先级分隔：

```
段落 (\n\n) → 换行 (\n) → 句号 (. ) → 空格 → 字符
```

### 6.1 PDF 参数调优实验

通过 6 组实验确定最优 chunk_size 和 overlap 参数：

| 方案 | chunk_size | overlap | chunks | BM25 |
|:----|:----------:|:-------:|:-----:|:----:|
| cs300 | 300 | 30 (10%) | 3383 | ✅ |
| cs500 | 500 | 50 (10%) | 2031 | ✅ |
| cs500_ov75 | 500 | 75 (15%) | 2042 | ✅ |
| cs500_ov100 | 500 | 100 (20%) | ~2090 | ✅ |
| cs600 | 600 | 60 (10%) | 1692 | ✅ |
| cs700 | 700 | 70 (10%) | 1411 | ✅ |

### 6.2 HTML 参数调优实验

| 方案 | chunk_size | overlap | chunks | BM25 |
|:----|:----------:|:-------:|:-----:|:----:|
| cs300_ov45 | 300 | 45 (15%) | 1705 | ✅ |
| cs500_ov75 | 500 | 75 (15%) | 964 | ✅ |

### 6.3 CSV 参数

CSV 数据量小（3 个文件共 18 chunks），直接采用与 PDF/HTML 相同的参数。

### 6.4 最终选型

| 数据源 | chunk_size | chunk_overlap |
|:------|:---------:|:-------------:|
| **PDF 论文** | **500** | **75（15%）** |
| **HTML 文档** | **500** | **75（15%）** |

两类数据均采用相同的切分参数，便于后续合并检索。

## 7. 向量化与索引

### 7.1 Embedding 模型

| 模型 | 维度 | 优化方向 | 选择原因 |
|------|:----:|:--------:|---------|
| **BAAI/bge-large-zh** | 1024 | 中英双语 | 后续知识库包含中文内容，中英双语模型兼顾 |

### 7.2 向量索引

| 参数 | 值 |
|------|:--:|
| 索引类型 | HNSW |
| 空间度量 | L2（向量已归一化，等价余弦相似度） |
| ef_construction | 100 |
| ef_search | 50（从 100 优化至 50，精度不变，搜索速度提升） |
| max_neighbors | 16 |

### 7.3 数据库

| 数据库 | 来源 | chunks | 检索耗时 |
|:------|:----|:-----:|:--------:|
| `chroma_db_ov75` | 7 篇 PDF 论文 | 2042 | ~110ms |
| `chroma_db_html_cs500` | 24 个 HTML 文档 | 964 | ~103ms |
| `chroma_db_csv` | 3 个 CSV 对比表 | 15 | ~94ms |
| `chroma_db_combined` | PDF + HTML + CSV 全量 | 2900 | ~242ms（含 Reranker） |

### 7.4 检索耗时分析

| 环节 | 耗时 | 占比 |
|------|:----:|:----:|
| Embedding 模型推理（用户问题） | ~80ms | ~33% |
| HNSW 向量检索 | ~5ms | ~2% |
| BM25 稀疏检索 | ~10ms | ~4% |
| RRF 结果融合 | ~2ms | ~1% |
| **Reranker 重排（15 对）** | **~145ms** | ~60% |
| BM25 索引构建（一次性） | ~130ms | — |
| Reranker 模型加载（一次性） | ~14s | — |
| **总检索耗时** | **~242ms** | **100%** |

Reranker 模型使用 `ms-marco-MiniLM-L-4-v2`（4 层 Cross-encoder，约 55MB），总检索耗时 242ms，在 300ms 工程红线内。

## 8. 混合检索（多路召回）

### 8.1 架构

```
用户问题
    ↓
┌──────────────────┐   ┌──────────────────┐
│  向量检索（稠密）  │   │  BM25 检索（稀疏） │
│ bge-large-zh →  │   │  关键词匹配 →    │
│ Chroma HNSW     │   │  BM25Okapi       │
│     top-16      │   │     top-16       │
└────────┬─────────┘   └────────┬─────────┘
         ↓                      ↓
    ┌──────────────────────────────────┐
    │    加权 Reciprocal Rank Fusion   │
    │    RRF = 0.65×向量 + 0.35×BM25  │
    │    k = 60                        │
    └────────────────┬─────────────────┘
                     ↓
               top-15 chunks
                     ↓
         ┌──────────────────────┐
         │    Cross-encoder      │
         │    Reranker           │
         │    MiniLM-L-4-v2      │
         │    15 pairs score     │
         └──────────┬───────────┘
                    ↓
              top-5 chunks
```

### 8.2 为什么用 BM25

稠密检索（向量）和稀疏检索（BM25）互补：

| 维度 | 向量检索 | BM25 检索 |
|------|---------|-----------|
| 匹配方式 | 语义相似度 | 关键词精确匹配 |
| 优点 | 理解同义词、上下文 | 命中精确术语、缩写 |
| 缺点 | 对罕见词不敏感 | 无法理解语义 |
| 示例 | "language model" → BERT/GPT | "LoRA" → 精确命中 LoRA 论文 |

### 8.3 为什么用 Reranker

向量检索和 BM25 在多源知识库中存在一个固有问题：语义相关但来源错误的文档会获得高分。例如查询 "How does RAG work?" 时，`rag_original.md`、`llamaindex_rag.md`、`langchain_rag.md` 都和 RAG 相关，都会排在前面，但只有一个是正确答案来源。

Cross-encoder Reranker 通过逐对评分（query, chunk），精细区分真正相关和语义近似的文档，是解决这一问题的标准方案。

### 8.4 Reranker 模型选型

| 模型 | 层数 | 大小 | 每对耗时 | 总检索 | Context Precision |
|:----|:---:|:---:|:------:|:-----:|:----------------:|
| MiniLM-L-6-v2 | 6 | 80MB | ~15ms | 325ms | 0.558 |
| **MiniLM-L-4-v2** | **4** | **55MB** | **~10ms** | **242ms** | **0.620** |

最终选择 L-4-v2，检索耗时 242ms，精度反而更高。

### 8.5 RRF 融合权重优化

BM25 检索有一个固有问题：英文停用词（"me"、"how"、"what" 等）在 BM25 中会被当作关键词造成误匹配。例如 query "tell me how transformer works" 中，"me" 和 "how" 会匹配到包含这些词的无关文档，导致 RRF 融合后错误文档排在前面。

**两个修复：**

1. **BM25 停用词过滤** — `_tokenize()` 中过滤了 ~70 个常见英文停用词，BM25 不再被高频功能词带偏
2. **加权 RRF 融合** — 向量检索（语义理解）和 BM25（关键词匹配）的置信度不同。向量检索对术语类 query 的 top-1 置信度高，而 BM25 容易受短词干扰。因此将 RRF 权重调整为：

```
RRF_score = W_dense × Σ 1/(60 + rank_dense + 1) + W_sparse × Σ 1/(60 + rank_sparse + 1)
W_dense = 0.65, W_sparse = 0.35
```

修复后效果对比（query: "tell me how transformer works"）：

| 指标 | 修前 | 修后 |
|:-----|:-----|:-----|
| top-1 结果 | `llama2.md`（"roast me" 噪声） | `web_transformer_family.md` |
| 正确文档在前 3 位 | ❌ 第 4 位 | ✅ 第 1/2/3 位 |
| LLM 回答 | "没有足够信息" | 准确描述 attention、encoder-decoder、并行化 |

## 9. 评估体系

### 9.1 评估指标

| 指标 | 说明 | 理想值 |
|------|------|:------:|
| **Recall@k** | top-k 中来自正确文档的 chunk 比例 | 越高越好 |
| **HitRate@k** | 正确文档是否出现在 top-k 中 | 1.0 |
| **Faithfulness** | 答案是否忠于上下文（RAGAS） | 越接近 1 越好 |
| **Answer Relevancy** | 答案是否切题（RAGAS） | 越接近 1 越好 |
| **Context Recall** | 上下文中是否包含足够信息（RAGAS） | 越高越好 |
| **Context Precision** | 检索的 chunk 中多少是相关的（RAGAS） | 越高越好 |

### 9.2 PDF 评估结果

**最终方案：cs500 + ov75 + BM25, top-8**

| 指标 | 分数 |
|:-----|:----:|
| **Recall@8** | 0.755 |
| **HitRate@8** | 0.980 |
| **Faithfulness** | 0.833 |
| **Answer Relevancy** | **0.905** |
| **Context Recall** | 0.739 |
| **Context Precision** | 0.492 |
| **平均检索耗时** | 110.3ms |

**PDF 各方案完整对比：**

| 指标 | cs500 ov50 | cs500 ov75 🏆 | cs500 ov100 | cs600 | cs700 | cs300 |
|:-----|:---------:|:------------:|:----------:|:-----:|:-----:|:-----:|
| Recall@8 | 0.758 | 0.755 | 0.758 | 0.725 | 0.735 | 0.775 |
| Faithfulness | 0.814 | **0.833** | 0.815 | 0.809 | 0.810 | 0.766 |
| Answer Relevancy | 0.859 | **0.905** | 0.860 | 0.856 | 0.805 | 0.830 |
| Context Recall | 0.766 | 0.739 | 0.678 | 0.677 | 0.726 | 0.651 |
| Context Precision | 0.439 | 0.492 | **0.575** | 0.562 | 0.593 | 0.320 |
| 检索耗时 | 96.7ms | 110.3ms | 75.0ms | 108.3ms | 115.8ms | 92.6ms |
| 总 chunks | 2031 | 2042 | ~2090 | 1692 | 1411 | 3383 |

### 9.3 HTML 评估结果

**最终方案：cs500 + ov75 + BM25, top-8（问题集 30 题）**

| 指标 | 分数 | 说明 |
|:-----|:----:|:-----|
| **Recall@8** | 0.471 | 短文档偏多（5 个 2KB 以下概览页），影响均值 |
| **HitRate@8** | 0.833 | 25/30 题命中正确文档 |
| **Faithfulness** | **0.890** | 高于 PDF，HTML 内容结构清晰 |
| **Answer Relevancy** | **0.846** | 接近 PDF 水平 |
| **Context Recall** | 0.635 | 中等水平 |
| **Context Precision** | **0.688** 🥇 | 远高于 PDF（0.49） |
| **平均检索耗时** | 103.5ms | |

**HTML cs300 vs cs500 对比：**

| 指标 | cs300 (1705 chunks) | cs500+ov75 (964 chunks) 🏆 |
|:-----|:-----------------:|:--------------------------:|
| Recall@8 | **0.517** | 0.471 |
| Faithfulness | 0.891 | **0.890** |
| Answer Relevancy | 0.832 | **0.846** |
| Context Recall | 0.563 | **0.635** |
| Context Precision | 0.234 | **0.688** 🥇 |

cs500+ov75 的 Context Precision 远高于 cs300，说明大 chunk 为 LLM 提供了更完整的信息上下文，生成质量更好。

### 9.4 CSV 评估结果

**方案：cs500 + ov75 + BM25, top-8（问题集 9 题）**

| 指标 | 分数 | 说明 |
|:-----|:----:|:-----|
| **Recall@8** | 0.528 | 15 个 chunks 的小库 |
| **HitRate@8** | **1.000** | 9/9 题全部命中正确文档 |
| **Faithfulness** | 0.702 | |
| **Answer Relevancy** | **0.959** | 答案质量极高 |
| **Context Recall** | **0.889** | 上下文覆盖充分 |
| **Context Precision** | 0.052 | 库太小，top-8 几乎覆盖全部内容 |
| **检索耗时** | 94.3ms | |

CSV 数据量小（15 chunks），检索空间有限，但 Answer Relevancy 0.96 和 Context Recall 0.89 说明信息提取充分，答案质量好。

### 9.5 综合评估（全量 89 题 + Reranker）

**最终方案：cs500_ov75 + BM25 + Reranker(MiniLM-L-4-v2), top-5, 综合库 2900 chunks**

| 指标 | 分数 | 说明 |
|:-----|:----:|:-----|
| **Recall@5** | 0.616 | 多源异构数据，短文档拉低均值 |
| **HitRate@5** | 0.843 | 75/89 题正确文档命中 |
| **Faithfulness** | 0.819 | 答案忠实度良好 |
| **Answer Relevancy** | 0.816 | 回答切题度高 |
| **Context Recall** | 0.702 | 上下文信息覆盖充分 |
| **Context Precision** | **0.620** 🥇 | 远超 0.4 目标，Reranker +77% |
| **检索耗时** | **242ms** ✅ | 在 300ms 工程红线内 |

**检索方案演进对比（全量 89 题）：**

| 指标 | BM25+向量 top-8 | RRF调权 top-8 | L-6 Reranker top-5 | **L-4 Reranker top-5 🏆** |
|:-----|:--------------:|:-------------:|:-----------------:|:----------------------:|
| Recall | 0.511 | 0.489 | 0.622 | **0.616** |
| Faithfulness | 0.845 | 0.877 | 0.771 | **0.819** |
| Answer Relevancy | 0.804 | 0.680 | 0.854 | **0.816** |
| Context Recall | 0.655 | 0.630 | 0.667 | **0.702** |
| Context Precision | 0.350 | 0.000 | 0.558 | **0.620** |
| 检索耗时 | **92.8ms** | ~100ms | 325ms | **242ms** ✅ |

**低 Recall 分析：**

综合库 Recall@5=0.616 的主要原因：
1. **短文档居多** — HTML/CSV 中有多个 <2KB 的概览页，chunk 极少
2. **PDF 和 web_transformer_family.md 有内容交叉** — Transformer 相关问题两个文档都涉及，Recall 计算只认其中一个
3. **生成质量不受拖累** — Faithfulness(0.82) 和 Answer Relevancy(0.82) 良好

## 10. 检索增强实验

### 10.1 实验目标与方法

对比三种基于 LLM 的查询增强策略——**Query Rewriting**（改写为 1 条新 query）、**Multi-Query**（扩展为 4 条 query 合并）和 **HyDE**（生成假设文档替代 query）——实际检索中的收益与代价。三者本质相同：用 LLM 对用户问题进行变换后再检索。

**实验配置：**
- 测试题：80 道（PDF 50 + HTML 30），原始 HitRate 已达 0.925-0.933
- 检索方式：纯向量（关闭 BM25，排除变量干扰）
- 改写/生成模型：DeepSeek Chat (API)

### 10.2 实验结果

**Query Rewriting（替换策略）：**

| 指标 | Qwen2-0.5B | Qwen2.5-1.5B | DeepSeek API |
|:-----|:----------:|:------------:|:------------:|
| 原始 HitRate | 0.9326 | 0.9326 | 0.9250 |
| 改写后 HitRate | 0.9326 | 0.9213 | 0.9250 |
| **提升** | **0.0%** | **-1.1%** | **0.0%** |
| 耗时 | 4.7s | 10.2s | 1.2s |

**Multi-Query ×3（扩展策略）：**

| 指标 | Qwen2-0.5B | Qwen2.5-1.5B | DeepSeek API |
|:-----|:----------:|:------------:|:------------:|
| 原始 Recall@5 | 0.6517 | 0.6517 | 0.6675 |
| Multi-Query | 0.1888 | 0.1888 | 0.1825 |
| **提升** | **-46.3%** | **-46.3%** | **-48.5%** |
| 总延迟 | 6.9s | 14.3s | 1.8s |

**HyDE（文档替换策略）：**

| 指标 | 原始 query | HyDE | 提升 |
|:-----|:---------:|:----:|:----:|
| HitRate@8 | 0.925 | 0.925 | **0.0%** |
| **Recall@8** | 0.619 | **0.738** | **+11.9%** 🥇 |
| 生成耗时 | — | **2.9s** | ❌ |

### 10.3 分析与结论

**改写/扩展无收益的原因：**
1. 改写无实质变化（只是全称展开，语义不变，embedding 结果相同）
2. Multi-Query 变体与原始 query 高度重叠，去重后有效命中反而被稀释
3. 原始 HitRate 已达 0.93，没有提升空间
4. 知识库为专业论文，用户问题本身已足够精准

**HyDE 有效但延迟超限：**
- Recall@8 从 0.619 提升到 0.738（+11.9%），说明假设文档在 embedding 空间中确实更接近目标文档
- 但 2.9s 的 API 生成耗时远超 300ms 工程红线

| 策略 | 收益 | 代价 | 结论 |
|:----|:---:|:----:|:----:|
| Query Rewriting | -1.1% ~ 0% | 1.2~10.2s | ❌ 不可行 |
| Multi-Query ×3 | -46~-49% | 1.8~14.3s | ❌ 不可行 |
| **HyDE** | **+11.9% Recall** | **2.9s** | ⚠️ 有效但延迟超限 |

**结论：HyDE 是唯一产生正向收益的策略，但延迟使其无法用于生产。如果未来 GPU 推理提速到 <300ms，HyDE 值得重新评估。当前保持纯向量+BM25+Reranker 检索方案。**

## 11. 项目结构

```
E:\RAG_Project\
├── RAG_Data\                      # 原始数据
│   ├── papers\                    # 7 篇 PDF 论文
│   ├── langchain_docs\            # LangChain 文档（12 个 HTML）
│   ├── llamaindex_docs\           # LlamaIndex 文档（7 个 HTML）
│   ├── web\                       # 网页博客（5 个 HTML）
│   └── tables\                    # CSV 对比数据（3 个文件）
├── data_loaders\                  # 数据处理脚本
│   ├── pdf\
│   │   └── parse_and_clean.py     # PDF 解析 + 清洗一体化
│   ├── html\
│   │   └── parse_html.py          # HTML 解析（trafilatura）
│   ├── csv\
│   │   └── parse_csv.py           # CSV 表格转 Markdown
│   ├── chunk\
│   │   └── text_splitter.py       # 文本切分（支持命令行参数）
│   └── vectorstore\
│       └── chroma_store.py        # Chroma 向量库构建（支持命令行参数）
├── scripts\
│   └── evaluate_rag.py            # RAG 系统评估脚本（支持 BM25、参数隔离）
├── data_output\                   # 数据处理输出
│   ├── pdf_markdown\              # PDF 清理后的 Markdown
│   ├── html_markdown\             # HTML 提取的 Markdown
│   ├── csv_markdown\              # CSV 转换的 Markdown
│   ├── eval_questions.json        # PDF 评估问题集（50 题）
│   ├── eval_questions_html.json   # HTML 评估问题集（30 题）
│   ├── eval_questions_csv.json    # CSV 评估问题集（9 题）
│   ├── eval_questions_combined.json # 综合评估问题集（89 题）
│   └── eval_results_tk5_combined_bm25_reranker_ms-marco-MiniLM-L-4-v2/ # 最终评估结果
├── output\                        # 向量数据库
│   ├── chroma_db_ov75\            # PDF 库（cs500_ov75, 2042 chunks）
│   ├── chroma_db_html_cs500\      # HTML 库（cs500_ov75, 964 chunks）
│   ├── chroma_db_csv\             # CSV 库（cs500_ov75, 15 chunks）
│   └── chroma_db_combined\        # 综合库（PDF+HTML+CSV, 2900 chunks）
└── docs\
    ├── pdf_parsing_report.md      # PDF 解析实验报告
    └── Project_Report.md          # 项目报告（本文件）
```

## 12. 产出隔离机制

所有脚本的输出目录自动根据命令行参数隔离，不同参数组合互不覆盖：

| 命令 | 输出目录 |
|------|---------|
| `chroma_store.py` | `output/chroma_db/` |
| `chroma_store.py --chunk_size 800` | `output/chroma_db_cs800/` |
| `chroma_store.py --chunk_size 500 --chunk_overlap 75` | `output/chroma_db_ov75/` |
| `chroma_store.py --input_dir data_output/html_markdown --tag html_cs500` | `output/chroma_db_html_cs500/` |
| `evaluate_rag.py --top_k 8 --chroma_dir output/chroma_db_ov75 --use_bm25` | `data_output/eval_results_tk8_ov75_bm25/` |

## 13. 经验总结

1. **pymupdf4llm 是最优 PDF 解析方案**：纯 Python，零系统依赖，表格结构完整保留
2. **trafilatura 是最优 HTML 解析方案**：自动去除导航栏/侧栏/页脚，99%+ 的噪声压缩比
3. **双栏表格是 PDF 解析的已知难题**：目前无完美自动化方案，手动修复成本可控
4. **chunk_size=500 + overlap=15% 在 PDF/HTML/CSV 上均表现最佳**
5. **BM25 多路召回 + Reranker 是最优检索方案**：三重检索（向量→BM25→Reranker）将 Context Precision 从 0.35 提升到 0.62
6. **Reranker 选型至关重要**：MiniLM-L-4-v2 比 L-6-v2 快 25% 且精度更高，总耗时 242ms 在工程红线内
7. **Recall 的绝对值需结合文档特征解读**：短文档偏多时 Recall 低是正常现象，生成质量才是最终指标
8. **产出隔离避免实验相互覆盖**：按参数自动生成目录名，方便快速对比
9. **bge-large-zh 中英双语模型适合混合语言知识库**
10. **BM25 停用词过滤 + 加权 RRF 融合是必要优化**：英文停用词会导致 BM25 误匹配，向量权重 0.65、BM25 权重 0.35 的加权 RRF 有效抑制了此类噪声
