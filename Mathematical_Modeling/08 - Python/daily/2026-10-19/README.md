---
title: Day49 - 微分方程建模 SEAR 谣言恐慌（代码）
type: code-readme
subject: mathematical-modeling
level: L3 经典数学模型（微分方程）/ L6 优化（参数反演）
tags:
  - mathematics
  - mathematical-modeling
  - differential-equation
  - ode
  - parameter-estimation
  - sensitivity-analysis
created: 2026-10-19
---

# Day49 · 微分方程建模：SEAR 恐慌传播（代码）

教案：[[Day49 - 微分方程建模SEAR谣言恐慌与县域应急撤离]]

## 文件

| 文件 | 说明 |
| --- | --- |
| `fear_spread_sear.py` | 主脚本：SEAR 模型积分 + 策略对比 + 参数反演 + 敏感性 + 决策曲线 |
| `fear_spread_sear.png` | 四联图（四仓室演化 / 策略对比 / 反演拟合 / 决策曲线） |

## 场景

云溪县化工厂爆燃，一条"有毒气体扩散"的群聊消息在县城传播。县应急办 48 小时内要定：要不要广播、什么时候、投多强。

## 10 问速答

1. **解决什么问题**：突发事件下"恐慌人群"随时间的演化，以及干预强度的定量决策。
2. **什么时候用**：有机制知识（谁影响谁）、但没有历史同期数据可学（ARIMA/ML 失效）。
3. **为什么适合**：传播是"散布者 × 未听说者"的**乘法**结构，天然是 ODE；可控量（广播强度）能直接进方程。
4. **数学模型**：SEAR（S/E/A/R 四仓室，$A$ = 恐慌并散布），$\dot A = \gamma E - (\omega+g)A$。
5. **求解方法**：`scipy.integrate.solve_ivp`（LSODA，刚性友好），`max_step=0.02` 保精度。
6. **参数从哪来**：$\beta$ 反演（`least_squares` + log 残差 + 多起点），$\gamma,\omega$ 取文献/现场经验。
7. **结果怎么读**：峰值 29.8% @ 第 10.3 天；最终累计 98.7% —— 两个数用途完全不同。
8. **怎么检验**：总量守恒 + 单调性 + 峰值条件 $\gamma E=(\omega+g)A$ + 拟合残差 + 可辨识性扫描。
9. **敏感性**：$\beta\pm25\%$ → 峰值 23.2%~34.3%，资源需多备 48%。
10. **坑**：分批投放反而更差（$E$ 库存照常补货）；参数拟合好 ≠ 参数可信（$\beta,\gamma$ 部分抵消）。

## 关键实跑数字

- 无干预峰值 **89,338 人（29.8%）@ 第 10.3 天**，$R_0=1.29>1$ 自我维持；
- 持续广播 → 峰值 **10,585（3.5%）**，降幅 88.2%；
- 分批投放（总剂量相同）→ 52,110（17.4%），**比持续更差**；40% 警力公平对比亦如此（41,660 vs 52,110）；
- 反演 $\hat\beta=2.15$（真值 2.20）、$\hat\gamma=1.26$（真值 1.20），拟合误差 3.9%；
- 决策曲线：$g\ge0.80$/天 才能把峰值压到 5%。

## 运行

```bash
cd /home/ubuntu/Note && ./.venv/bin/python "Mathematical_Modeling/08 - Python/daily/2026-10-19/fear_spread_sear.py"
```

全部断言通过；`-W all` 无告警；已注册中文字体避免乱码。
