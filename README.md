# 表白墙数据分析 (NBFLS BBS Analysis)

本项目对宁波外国语学校表白墙 QQ 空间 2022.05.30 至 2026.04.04 的投稿图片进行 OCR 识别、文本清洗、大模型语义提取和可视化分析，最终呈现校园社交行为的统计特征。

>  **声明**：所有公开发布的数据均已脱敏，姓名仅保留拼音首字母，不含原始帖文内容，仅呈现群体统计结果。

## 数据处理流程

1. **图片下载** – 使用 [QQ空间导出助手](https://github.com/ShunCai/QZoneExport) 抓取图片。
2. **OCR 识别** – [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 提取文字，多线程并行处理（`parse_image.py`）。
3. **文本清洗** – 去除 UI 垃圾、修正 OCR 黑话、过滤无效内容（`clean_jsonl.py`）。
4. **特征提取** – 调用[硅基流动](https://cloud.siliconflow.cn/) DeepSeek V4 Flash 大模型解析帖子，提取发帖人/目标对象班级、学部、匿名状态、目的、情感等字段（`extract_feature.py`）。
5. **数据分析** – Jupyter Notebook 完成统计与可视化，输出图表与词云。

## 仓库结构

> **受Jupyter Notebook 环境要求与Paddle OCR推荐Python版本不一致影响，数据分析Notebook的环境与项目其他文件的环境不一致**

```
.
├── data/
│   ├── raw_image/          # 原始图片（不公开）
│   ├── text/               # OCR识别后的原始文件与初步处理文件（不公开）
│   └── feature_data/       # 提取的结构化 CSV（公开版已脱敏）
├── scripts/                # 处理脚本
│   ├── parse_image.py
│   ├── clean_jsonl.py
│   ├── extract_feature.py
│   └── data_masking.py
├── notebook/
│   ├── requirements.txt    # Jupyter Notebook依赖
│   └── bbs_analysis.ipynb  # 数据分析与可视化
├── README.md
└── requirements.txt        # 项目其他部分 Python 依赖
```

## 使用方法

### 环境配置

项目数据分析以外部分推荐使用 Python 3.9，数据分析部分使用3.10+，安装对应依赖：
```bash
pip install -r requirements.txt
```

OCR 需要单独安装 PaddlePaddle 和 PaddleOCR，请参考 [官方文档](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/quickstart.md)。

### 运行步骤

1. **OCR 识别**  
   ```bash
   python scripts/parse_image.py          # 默认4线程
   python scripts/parse_image.py 8        # 指定线程数
   ```

2. **文本清洗**  
   ```bash
   python scripts/clean_jsonl.py
   ```

3. **大模型提取特征**（需设置环境变量 `DEEPSEEK_API_KEY`）  
   ```bash
   export DEEPSEEK_API_KEY="你的API密钥"
   python scripts/extract_feature.py
   ```

4. **脱敏处理**（如需要）  
   ```bash
   python scripts/data_masking.py
   ```

5. **可视化分析**  
   在 Jupyter Notebook 中打开 `notebook/analysis.ipynb` 逐步执行。

## 主要发现

- 发帖目的以**表白**、**寻人**、提问为主，三者占总数的约60%。
- 匿名率整体较高，表白类匿名比例 > 80%，祝福、寻物类实名相对高；所有类别实名均未过半。
- 初中部（J）学生发帖和被提及频次断档最高，普高部（S）次之，国际部（I）最少。
- 帖子量在2022年中呈现显著高峰，后迅速下降稳定。
- 部分人名被提及的次数断档领先。
- 祝福对象除了祝老师新婚快乐以外，以考试顺利（尤其是中考）为主。
- 我校表白墙发帖人颜控为主。

## 局限与说明

- 大模型提取的特征未经过人工校验，可能存在分类偏差。
- 部分图片 OCR 效果不佳，少量帖子遗漏。
- 代码部分由 AI 辅助生成，结构不规范。
- 项目仅反映单个学校表白墙的数据，不具备普遍性。

## 致谢
感谢硅基流动赠送的27元优惠券
感谢DeepSeek开源了V4 Flash 与 Pro模型
感谢QQ空间导出助手的作者
感谢我自己
  
谴责网页版千问
谴责Gemini 3.1 Pro

## License

本项目代码采用 [MIT License](LICENSE)。数据部分仅限于学术研究。