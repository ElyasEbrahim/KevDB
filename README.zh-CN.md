# KevDB（中文简介）

[English](README.md) · 完整文档、实验记录和代码注释都是英文，这里只做简短介绍。

**在 DuckDB 里用开源决策模型执行语义 SQL。**

决策模型不生成文字：它读一条记录，回答带类型的问题（是非、单选、打分），并给出概率。这让它天然适合做数据库算子：
每行一次便宜、确定的调用，概率可以设阈值、校准和聚合。

最有名的决策模型 Jev（TypeSafe AI）是闭源的，只能通过付费接口调用。现在已经有几个开源模型接受同样的请求格式、能在笔记本上运行：
Kev（0.8B、4B、9B）、Clef-flash 和 Laya。本项目把它们接进 DuckDB，作为 SQL 函数使用，并按事先写定的标准检验：
它们离 Jev 还差多少，以及本机模型能做到哪些云端接口做不到的事。

## 主要结论

- **开源模型还不能原样替代 Jev。** Kev-4B、Kev-9B、Clef-flash 在影评上与 Jev 相当，在餐馆评论的方面题上都不及；商品匹配上只有 Kev-9B 相当。
- **差距在概率的刻度，不在排序。** 用 200 条人工标注重新校准后，三个模型与 Jev 的差距缩到 1.4 个百分点以内。
- **本机模型独有的好处是复用读过的正文。** 长文档上反复换题时，Kev-4B 快 43 倍、Kev-0.8B 快 23 倍，答案逐位相同。
- **当语义 SQL 的后端**，Kev-4B 在 SemBench 电影场景的 10 条查询上平均质量 0.803（Jev 0.765），不花钱，但耗时是 Jev 的 8.7 倍。
- **校准参数可以存进数据库**，由 SQL 宏应用；它能修刻度，修不了排序。
- **语义连接的问法**：把候选打包进一个请求，在 Jev 上省九成请求且不降质量，在 Kev-4B 上只快 1.7～1.9 倍；换一种连接谓词，打包会改变回答。
- **微调 Kev 能否追上 Jev**：实验已设计好，尚未运行。

## 快速开始

```bash
git clone https://github.com/ElyasEbrahim/KevDB.git
cd KevDB
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q             # 离线测试，不需要密钥和模型
./scripts/open_models.sh setup            # 在 open_models/ 里安装 Kev 和 Clef-flash（约 49 GB，只支持 Apple Silicon）
./scripts/open_models.sh serve kev-4b     # 在 127.0.0.1:8009 起服务，运行在沙盒里，不能联网
```

用法、全部结果和复现步骤见英文的 [README](README.md)、[docs/](docs/) 和 [experiments/](experiments/)。

## 许可证

MIT。本项目与 TypeSafe AI 以及 Kev、Clef、Laya 的作者均无关联；没有用 Jev 的输出训练或校准任何开源模型。
