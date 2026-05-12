# V2 设计案

**版本**：v2  
**创建日期**：2026-04-25  
**工作区**：~/regression-lgbm-v2  
**环境配置**：pip install pandas numpy lightgbm pytest baostock qlib

---

## 如何运行

一键运行五周回测（最常用）
cd ~/regression-lgbm-v2
python src/run_weekly_test.py

运行完成后会输出：
控制台：回测成绩单
output/result.csv：最终提交文件（Top-5 股票 + 权重）
output/weekly_report.csv：详细成绩
logs/：完整运行日志

本地验证提交文件是否合法
当比赛方下发 test.csv 后，把它放入 data/ 目录，再运行：
python score_self.py
会直接输出本次提交的加权收益率得分。

运行所有单元测试
python -m pytest tests/ -v


## 核心改进（相比V1）

| 改进点 | V1 | V2 |
|--------|----|----|
| 代码结构 | 单文件 | 模块化（每职责一个文件） |
| 数据管理 | 纯pandas | qlib数据层 |
| 特征 | 纯技术指标（21个） | 技术指标 + 基本面（PE/PB/PS/PCF） |
| 输入校验 | 无 | 每模块有断言 |
| 日志 | print | 结构化日志文件 |
| 测试 | 无 | 单元测试 + 集成测试 |

---


## 关键决策记录

### 决策① qlib使用范围
**选择**：qlib管理数据层 + 特征表达式，模型层自定义  
**理由**：评分接口（score_self.py）固定，完全迁移qlib风险高；
qlib表达式引擎提供更丰富特征是核心价值，模型自定义便于实验对比。

### 决策② 数据来源
**选择**：价格数据来自stock_data.csv；基本面（PE/PB/PS/PCF）从Baostock拉取后缓存本地  
**理由**：价格数据已有无需重复拉取；基本面数据必须从外部获取；
缓存避免重复请求，可手动检查数据质量。  
**依赖**：需要baostock >= 0.9.1

### 决策③ 负PE处理
**选择**：拆成两个特征，不丢弃任何信息  
is_profitable = (peTTM > 0).astype(float) # 0或1 earnings_yield = 1.0 / peTTM # 负值保留
**理由**：负PE代表公司亏损，是重要信息；
用1/PE替代PE本身可以收敛数值范围，避免极端值；
同理适用于PB/PS/PCF。

### 决策④ 基本面数据日频对齐
**选择**：前向填充 + 额外增加 days_since_report 特征  
days_since_report = (当前日期 - 上次季报发布日).days
范围：0（刚发布）~ 90（快到下次季报）
**理由**：前向填充符合实际投资逻辑（季报发布前用上期数据）；
不修改PE等原始值的大小，而是让模型通过days_since_report
自动学习数据新鲜度对预测的影响，比手动加权更客观。

### 决策⑤ qlib_dumper 实现方式（含踩坑记录）

**问题过程**：
- v1 手写 .bin，文件名用 `close.bin` → D.features() 静默返回空
- v2 改用 DumpDataAll → pip 版 qlib 无 scripts 模块，无法导入
- v3 回到手写 .bin，从 qlib 源码 file_storage.py 第289行确认
  正确文件名格式为 `close.day.bin`，修复后 20/20 通过

**最终选择**：手写 .bin（v3），文件名格式 `{field}.{freq}.bin`

**根本原因**：Linux 文件系统大小写敏感 +
             qlib 期望文件名含频率后缀（.day.）
             两个细节缺一不可

### 决策⑥ technical.py 表达式踩坑记录

**问题1**：qlib 不内置 RSI 算子
  解决：用基础算子手动推导
  gain = (delta + Abs(delta)) / 2
  loss = (Ref(close,1) - close + Abs(delta)) / 2
  注意：不能用 -(Sub对象)，qlib不支持对复合表达式取负号
       必须调换减法顺序规避

**问题2**：qlib D.features() 返回的 index 顺序
  实际：(instrument, datetime)
  文档写的：(datetime, instrument)
  结论：以实际为准，测试用 set() 比较而非顺序比较

**问题3**：MACD柱状图精度
  qlib 对相同子表达式独立计算两遍，float32 累积误差在高价股上明显
  解决：改用相对误差验证（<50%），而非绝对误差（<0.01）
               
---


## 模块契约

### 模块A：loader.py
- **输入**：CSV路径（str）
- **输出**：DataFrame，列 stock_code(str,6位) / date(datetime) / open/close/high/low(float>0) / volume / turnover
- **断言**：股票数量在295~305之间；每只股票行数>100；无NaN

### 模块B：fetcher.py
- **输入**：stock_code列表，start_date，end_date
- **输出**：DataFrame，列 stock_code / date / peTTM / pbMRQ / psTTM / pcfNcfTTM
- **断言**：列存在；日期范围合理；结果缓存到 data/fundamental/

### 模块C：qlib_dumper.py
- **输入**：模块A输出的DataFrame，qlib数据存放路径
- **输出**：磁盘上qlib格式数据
- **断言**：转换后qlib可正常init；D.features()可正常查询

### 模块D：technical.py
- **输入**：qlib环境，stock_list，date_range
- **输出**：DataFrame，index=(date,stock_code)，21个技术特征列
- **断言**：RSI在0~100；无Inf值；无全NaN列

### 模块E：fundamental.py
- **输入**：模块B输出，需要对齐的日期列表
- **输出**：DataFrame，index=(date,stock_code)，列 is_profitable / earnings_yield / pb_inv / ps_inv / pcf_inv / days_since_report
- **断言**：填充后NaN比例<20%；days_since_report在0~120之间

### 模块F：normalizer.py
- **输入**：DataFrame（含特征列），待标准化的列名列表
- **输出**：同结构DataFrame，特征值已截面Z-Score标准化
- **断言**：标准化后每列每天均值≈0；不修改label列

### 模块G：lgbm_regressor.py
- **输入（fit）**：X_train(DataFrame), y_train(Series)
- **输出（predict）**：Series，index与X_test相同
- **断言**：训练前校验列名；预测前校验列名一致；输出长度==len(X_test)

### 模块H：top_k.py
- **输入**：pred_score(Series，index=stock_code)，k=5
- **输出**：DataFrame，列 stock_code / weight
- **断言**：weight之和<=1.0；所有weight>0；行数<=5

### 模块I：scorer.py
- **输入**：top_k结果，测试周起止日期，原始数据
- **输出**：float（加权收益率得分）
- **断言**：result.csv写入前校验；得分=-999时报异常

---

## 日志规范
logs/run_YYYYMMDD_HHMMSS.log 格式：[时间] [级别] [模块名] 消息内容 级别：INFO / WARN / ERROR

---

## 开发顺序
第一阶段（数据层）： Step 1: loader.py + test_loader.py  Step 2: fetcher.py  Step 3: qlib_dumper.py 

第二阶段（特征层）： Step 4: technical.py Step 5: fundamental.py Step 6: normalizer.py + test_features.py

第三阶段（模型+策略层）： Step 7: lgbm_regressor.py + test_model.py Step 8: top_k.py + test_strategy.py

第四阶段（集成）： Step 9: scorer.py Step 10: run_weekly_test.py Step 11: 五周测试，对比V1 

---


##  踩坑与修复记录
坑①：qlib_bin 目录结构错误
问题：qlib_dumper 写入时文件名缺少 .day. 后缀，且目录结构反了
实际写成：features/close/sh600519.bin
qlib 期望：features/sh600519/close.day.bin
修复：删除旧数据，重新 dump，得到正确结构
根因：Step 3 测试时用的是临时目录，实际生产目录从未被干净初始化

坑②：instruments/all.txt 大小写问题
问题：all.txt 里股票代码大小写与 features/ 目录不一致
修复：经验证 qlib 内部做了大小写不敏感处理，实际无需修改

坑③：qlib_data 目录为空
说明：data/qlib_data/ 是最初规划时的预留目录，实际数据在 data/qlib_bin/，无害，无需处理

坑④：src/evaluate/ 目录为空
说明：原计划存放 IC、Sharpe、最大回撤等评估工具，对比赛无直接影响，暂不填充

坑⑤：fetch() 参数名错误
问题：主程序调用时写的是 stock_code_list=，实际参数名是 stock_codes=
修复：改正参数名

坑⑥：stock_data.csv 股票代码是 int64
问题：pandas 读取后 000001 变成整数 1，前导零丢失
修复：全流程统一加 .astype(str).str.zfill(6)
同时修复：top_k.py 输出时加补零，read_csv 时加 dtype={'股票代码': str}

#	Bug	影响	修复位置
修复的 5 个核心 Bug
Bug1	标签泄漏	训练集最后5行的 label 用到了测试周价格	run_weekly_test.py：训练集截止 safe_cutoff = cutoff - 5交易日
Bug2	基本面 index 大小写不一致	technical 输出 sz000001，fundamental 输出 SZ000001，join 后基本面全为 NaN	run_weekly_test.py：join 前统一 lowercase_index()
Bug3	ffill 跨股票污染	直接对 MultiIndex 做 ffill，A股数据填到B股	run_weekly_test.py：改为 groupby(level='instrument').ffill()
Bug4	turnover 字段缺失	qlib_bin 里没有 turnover.day.bin	重新干净 dump，现已包含所有字段
Bug5	scorer.py 用收盘价	与官方 score_self.py 用开盘价口径不一致	scorer.py：close → open

*最后更新：2026-04-29*



