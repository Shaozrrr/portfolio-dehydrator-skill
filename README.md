# 持仓脱水器 / Portfolio Dehydrator

在波动与叙事交织的 Web3 市场里，风险从来不会因为持仓数量变多而自动分散，更多时候，它只是换了一种更隐蔽的方式累积。`持仓脱水器` 是一个面向 Web3 投资者的量化持仓诊断与仓位优化 skill，用于识别组合中的冗余暴露、下行性价比偏弱资产和隐藏回撤风险，并给出更有依据的配置建议。

`Portfolio Dehydrator` is a quantitative portfolio diagnosis and allocation optimization skill for Web3 investors. It helps expose hidden overlap, compress unproductive volatility, and rebuild a cleaner, more defensible allocation structure using evidence-based risk controls.

## What It Does

- 识别组合里“看似分散、实则高度重叠”的风险暴露
- 用 `Sortino`、`Calmar`、`最大回撤` 和相关性，而不是单看涨跌幅做评估
- 在严格仓位约束下，输出可执行的优化比例和客户版分析报告
- 对中国大陆常见网络环境做容错：优先 `OKX`，必要时自动降级 `Gate.io`、`Bybit`、`Bitget`
- 在 API 不稳定或单个币种数据异常时，尽量保留部分真实分析结果；若真实数据不足，则跳过该资产而不是伪造 Mock 数据
- 用“样本分层 + 数据置信度”解释结果边界，让客户知道哪些结论更稳、哪些只适合参考
- 提供简单的组合压力测试，帮助客户理解在几种常见回撤情景下的潜在损失边界

## Why It Is Different

大多数 Web3 持仓工具更关注盈亏追踪，而这个 skill 更关注三个问题：

1. 你的仓位到底是不是“真分散”
2. 你承担的波动，值不值得
3. 哪些仓位应该保留，哪些仓位应该让位给更高质量的配置或现金缓冲

它不是一个“预测下一只暴涨币”的工具，而是一套更偏风险收益效率的配置决策框架。

## Decision Framework

这个 skill 的底层逻辑基于现代投资组合理论，并做了更适合 Web3 持仓场景的改造：

- `相关性矩阵`：识别高重叠资产对，避免“假分散”
- `Sortino Ratio`：衡量下行风险调整后的收益效率
- `Calmar Ratio`：衡量收益相对于最大回撤是否划算
- `Maximum Drawdown`：直接衡量组合或单资产的底线风险
- `约束优化`：通过 `scipy.optimize` 的 `SLSQP` 在严格上限下分配权重

当前版本的公共约束包括：

- `BTC`、`ETH`：单币上限 `50%`
- 主流蓝筹：单币上限 `30%`
- 长尾 / Meme：单币上限 `15%`
- 数据不足的新币：单币上限 `5%`

如果某个资产在最近样本里表现出：

- 下行性价比偏弱
- 最大回撤过深
- 与更优资产高度相关

系统会进一步收紧它的有效上限，优先把仓位让给更高质量资产或 `USDT` 缓冲。

## Inputs

当前后端支持以下输入形式：

- 代币列表：`BTC ETH SOL PEPE`
- 带权重的自然语言持仓：`我现在 40% BTC、30% ETH、20% PEPE、10% USDT`
- 结构化参数：
  - `tokens: list[str]`
  - `total_capital: float`
  - `current_weights: dict[str, float]`

说明：

- 如果用户提供了原始持仓比例，报告会按真实持仓做“优化前 vs 优化后”对比
- 如果没有提供原始比例，系统默认以风险资产等权持有作为参考基线
- 截图 OCR、地址解析、链上资产聚合建议由上游产品先完成，再把标准化后的 token 和权重传给这个后端

## Outputs

生成结果是客户可读的中文 Markdown 报告，通常包含：

- 执行摘要
- 资产两两相关系数表
- 单资产画像表
- 风险重叠结论
- 优化前 vs 优化后对比
- 最终建议配置
- 调仓顺序
- 合规与风险揭示

## Example Use Cases

- “帮我看看 `BTC ETH SOL PEPE ARB` 有没有重复暴露”
- “我现在 `35% BTC、25% ETH、20% LINK、10% ARB、10% USDT`，怎么优化”
- “给客户出一份更专业的持仓诊断报告，不要只有一个结果”

## Repository Structure

```text
web3-portfolio-optimizer/
├── SKILL.md
├── README.md
├── agents/
│   └── openai.yaml
├── assets/
│   ├── requirements.txt
│   └── web3_portfolio_optimizer.py
└── references/
    └── implementation-spec.md
```

## Quick Start

### 1. Install dependencies

```bash
python3 -m pip install -r assets/requirements.txt
```

### 2. Run the bundled backend

```bash
python3 assets/web3_portfolio_optimizer.py --tokens BTC,ETH,PEPE,ARB --capital 10000
```

### 3. Run with current holdings

```bash
python3 assets/web3_portfolio_optimizer.py \
  --tokens BTC,ETH,USDT,PEPE \
  --capital 10000 \
  --weights "BTC=40,ETH=30,USDT=20,PEPE=10"
```

## Validation

本 skill 当前已经通过本地结构校验：

```bash
python3 /Users/shaozhaoru/.codex/skills/.system/skill-creator/scripts/quick_validate.py .
```

## Security Notes

- 当前实现只访问公开市场数据接口，不要求钱包私钥
- 稳定币作为无风险基准与缓冲资产处理
- 历史数据分析不构成收益承诺
- 面向客户使用时，建议同步披露“历史表现不代表未来”

## Publishing Notes

- GitHub Repository: <https://github.com/Shaozrrr/portfolio-dehydrator-skill>
- 当前名称：`持仓脱水器 / Portfolio Dehydrator`
- 当前 skill 触发名：`web3-portfolio-optimizer`

## License

MIT
