# Resources 资料库

汇总课程、书籍、视频、博客、论文，按 Level / 问题类型分类。

每条记录：名称、难度（入门 / 进阶）、语言、推荐理由、链接。

- **可下载的文档**（论文、PDF、官方文档、可保存网页）统一下载到 `Downloads/` 子目录，索引记录本地路径，笔记引用本地文件而非外链；
- **仅在线资源**（视频、需登录等）才只记链接，并标注"仅在线"。

建议标注状态：已读 / 在读 / 待读，并定期清理。

## 优化 / 分解（Level 6, 8）

- **Benders 分解入门（Cornell 计算优化开放教材）** — 入门，英文，[Optimization Wiki: Benders decomposition](https://optimization.cbe.cornell.edu/index.php?title=Main_Page)（**仅在线**：站点为动态 wiki，无稳定 PDF；本 README 2026-10-06 笔记引用时该条目页尚未创建，可用站内搜索 "Benders"）状态：待读
- **Benders, J. F. (1962). Partitioning procedures for solving mixed-variables programming problems.** *Numerische Mathematik* 4(1):238–252 — 进阶，英文，Benders 分解的原始论文（Springer，付费/需机构访问）（**仅在线**）状态：待读
- **Birge & Louveaux, *Introduction to Stochastic Programming*（第 2 版）** — 进阶，英文，两阶段随机规划与 L-shaped 方法（Benders 的随机规划专用版）的经典教材，第 5 章 L-Shaped Method 与本课 Day38 直接对应（**仅在线**，书籍）状态：待读

## 图论 / 路线规划（Level 7）

- **Edmonds, J. & Johnson, E. L. (1973). Matching, Euler tours and the Chinese postman.** *Mathematical Programming* 5(1):88–124 — 进阶，英文，把**中国邮递员问题归约到最小权完美匹配**的里程碑论文，是 Day41 理论的直接来源（Springer，付费/需机构访问）（**仅在线**）状态：待读
- **管梅谷（Mei-Ko Kwan, 1962）中国邮递员问题** — 入门，中文，CPP 的原始提出（《数学学报》/经典运筹学教材均收录）。原始中文文献检索受限，详见各运筹学教材「中国邮递员问题」章节（**仅在线**）状态：待读
- **Wikipedia: Route inspection problem（中国邮递员问题）** — 入门，英文，[Route inspection problem](https://en.wikipedia.org/wiki/Route_inspection_problem)（**仅在线**：Wiki 动态页面，无稳定 PDF；本 README 2026-10-10 笔记引用时宿主机无外网，未能抓取正文）状态：待读
- **Hierholzer (1873) 欧拉回路算法** — 入门，欧拉回路求解的经典算法，Day41 代码的 Step 4 实现依据（**仅在线**，教材内容）状态：待读

> ⚠️ 说明：2026-10-10 备课时宿主机**无外网出口**（`curl`/`web_fetch` 均超时），故本日资料**全部按"仅在线"登记、未下载 PDF**；待网络恢复后补下 Edmonds–Johnson 1973 与管梅谷 1962 原文。

## 元启发式 / 调度（Level 5–6）

- **Glover, F. (1986). Future paths for integer programming and links to artificial intelligence.** *Computers & Operations Research* 13(5):533–549；及 Glover (1989/1990) Tabu Search 系列（*ORSA Journal on Computing*） — 进阶，英文，**禁忌搜索（Tabu Search）的奠基论文**，是 Day42 理论的直接来源（Elsevier/INFORMS，付费/需机构访问）（**仅在线**）状态：待读
- **Lawler, E. L. (1977). A pseudopolynomial algorithm for sequencing jobs to minimize total tardiness.** *Annals of Discrete Mathematics* 1:331–342 — 进阶，英文，证明单机总延误 $1\|\sum T_j$ 的伪多项式 DP，Day42 复杂度与 NP-hard 论断的来源（Elsevier，付费）（**仅在线**）状态：待读
- **Rachamadugu, R. & Morton, T. E. (1982). Myopic heuristics for the weighted tardiness problem.** CMU 工作论文 — 进阶，英文，**ATC（Apparent Tardiness Cost）规则**的原始出处，Day42 构造规则对照的来源（CMU，可索取）（**仅在线**）状态：待读
- **Pinedo, M. *Scheduling: Theory, Algorithms, and Systems*（第 5 版）** — 进阶，英文，调度领域标准教材，第 3 章单机调度（EDD/WSPT/ATC 与总加权延误）与 Day42 直接对应（Springer，付费）（**仅在线**）状态：待读
- **Wikipedia: Tabu search** — 入门，英文，[Tabu search](https://en.wikipedia.org/wiki/Tabu_search)（**仅在线**：Wiki 动态页面；2026-10-11 备课时宿主机无外网，未能抓取正文）状态：待读

> ⚠️ 说明：2026-10-11 备课时宿主机**仍无外网出口**（`curl` 超时），故本日资料**全部按"仅在线"登记、未下载 PDF**；待网络恢复后补下 Glover 1986 与 Lawler 1977 原文。

## 群体智能 / 蚁群（Level 5–6）

- **Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization by a colony of cooperating agents.** *IEEE Transactions on Systems, Man, and Cybernetics, Part B* 26(1):29–41 — 进阶，英文，**蚁群算法 ACO / Ant System 的奠基论文**，是 Day43 理论的直接来源（IEEE，付费/需机构访问）（**仅在线**）状态：待读
- **Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: a cooperative learning approach to the traveling salesman problem.** *IEEE Transactions on Evolutionary Computation* 1(1):53–66 — 进阶，英文，**ACS（Ant Colony System）** 面向 TSP 的经典改进（引入伪随机比例选择与局部更新），Day43 落地增强的理论背景（IEEE，付费）（**仅在线**）状态：待读
- **Dorigo, M., & Stützle, T. *Ant Colony Optimization*（MIT Press, 2004）** — 进阶，英文，ACO 领域标准专著，覆盖 AS/ACS/MAX-MIN AS 各变体与 TSP/调度/路由应用，与 Day43 直接对应（MIT Press，付费）（**仅在线**）状态：待读
- **Wikipedia: Ant colony optimization algorithms** — 入门，英文，[Ant colony optimization algorithms](https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms)（**仅在线**：Wiki 动态页面，无稳定 PDF；2026-10-12 备课时宿主机外网受限，未能抓取正文）状态：待读

> ⚠️ 说明：2026-10-12 备课时宿主机**外网仍受限**（`web_search` 禁用、`web_fetch`/`curl` 多数超时；仅 arXiv 偶发 200 但无可下载目标），故本日资料**全部按"仅在线"登记、未下载 PDF**；待网络恢复后补下 Dorigo 1996/1997 与 Dorigo–Stützle 2004 专著。


## 群体智能 / 粒子群（Level 5–6）

- **Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.** *Proceedings of ICNN'95 — International Conference on Neural Networks*, vol. 4, pp. 1942–1948, IEEE — 进阶，英文，**粒子群优化 PSO 的奠基论文**，是 Day44 理论的直接来源（IEEE，付费/需机构访问）（**仅在线**）状态：待读
- **Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.** *1998 IEEE International Conference on Evolutionary Computation Proceedings*, pp. 69–73 — 进阶，英文，**引入惯性权重 $w$** 的关键改进，Day44 参数讨论（$w,c_1,c_2$）的直接依据（IEEE，付费）（**仅在线**）状态：待读
- **Clerc, M., & Kennedy, J. (2002). The particle swarm — explosion, stability, and convergence in a multidimensional complex space.** *IEEE Transactions on Evolutionary Computation* 6(1):58–73 — 进阶，英文，**收缩因子（constriction）** 与收敛性分析的经典论文，Day44 常用参数 $c_1=c_2\approx1.49445$ 的来源（IEEE，付费）（**仅在线**）状态：待读
- **Weiszfeld, E. (1937). Sur le point pour lequel la somme des distances de n points donnés est minimum.** *Tôhoku Mathematical Journal* 43:355–386 — 进阶，多设施 Weber 问题的**经典解析迭代算法**原始出处，Day44 中 k=1 裁判方法的来源（**仅在线**，历史文献）状态：待读
- **Wikipedia: Particle swarm optimization** — 入门，英文，[Particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization)（**仅在线**：Wiki 动态页面，无稳定 PDF；2026-10-13 备课时宿主机无外网，未能抓取正文）状态：待读

> ⚠️ 说明：2026-10-13 备课时宿主机**仍无外网出口**（`curl`/`web_fetch` 均超时），故本日资料**全部按"仅在线"登记、未下载 PDF**；待网络恢复后补下 Kennedy–Eberhart 1995、Shi–Eberhart 1998 与 Clerc–Kennedy 2002 原文。

## 进化算法 / 差分进化（Level 5–6）

- **Storn, R., & Price, K. (1997). Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces.** *Journal of Global Optimization* 11(4):341–359 — 进阶，英文，**差分进化 DE 的奠基论文**，是 Day45 理论的直接来源（Springer，付费/需机构访问）（**仅在线**）状态：待读
- **Price, K. V., Storn, R. M., & Lampinen, J. A. *Differential Evolution: A Practical Approach to Global Optimization*（Springer, 2005）** — 进阶，英文，DE 领域标准专著，覆盖 DE/rand/1/bin、DE/best、F/CR 参数调校与工程应用，与 Day45 直接对应（Springer，付费）（**仅在线**）状态：待读
- **Das, S., & Suganthan, P. N. (2011). Differential evolution: A survey of the state-of-the-art.** *IEEE Transactions on Evolutionary Computation* 15(1):4–31 — 进阶，英文，**DE 综述经典**，系统梳理各变异策略与自适应变体（含 JADE/SHADE），Day45 进阶路线的入口（IEEE，付费）（**仅在线**）状态：待读
- **MetaDE: Evolving Differential Evolution by Differential Evolution**（arXiv:2502.10470）— 进阶，英文，**已下载** PDF，用元进化自动搜索 DE 算法结构，Day45 「进化算法自调参」方向的现代延伸 → `Resources/Downloads/MetaDE-Evolving-Differential-Evolution-by-Differential-Evolution.pdf` 状态：待读
- **Differential evolution outside the box**（arXiv:2004.10489）— 进阶，英文，**已下载** PDF，讨论 DE 跳出常规（参数自适应/种群结构）的设计思想与收敛分析，Day45 参数 F/CR 讨论的深化 → `Resources/Downloads/Differential-Evolution-Outside-the-Box.pdf` 状态：待读
- **Wikipedia: Differential evolution** — 入门，英文，[Differential evolution](https://en.wikipedia.org/wiki/Differential_evolution)（**仅在线**：Wiki 动态页面，无稳定 PDF；2026-10-14 备课时 Wikipedia 超时，仅 arXiv 可达）状态：待读

> ⚠️ 说明：2026-10-14 备课时宿主机**外网部分受限**（Wikipedia 超时、arXiv API 时通时断，但 arXiv 主页与 PDF 下载可达），故**成功下载 2 篇 arXiv PDF**（MetaDE 2502.10470、DE outside the box 2004.10489）；Storn–Price 1997 奠基论文与 Das–Suganthan 2011 综述（Springer/IEEE 付费）**仅在线**登记，待网络/机构访问恢复后补下。

## 时间序列 / 预测（Level 4）

- **Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*.** Holden-Day（后由 Wiley 出版第 5 版, 2015）— 进阶，英文，**ARIMA / Box-Jenkins 方法论的奠基之作**，是本讲（Day46）「差分 → ACF/PACF 定阶 → 估参 → 诊断 → 预测」整套流程的原始来源（Wiley，付费/需机构访问）（**仅在线**）状态：待读
- **Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice*（第 3 版, 2021, OTexts）** — 入门，英文，**免费在线教材**，第 8 章 ARIMA 与本讲（Day46）逐节对应（含 ACF/PACF 判读、信息准则定阶、残差诊断、预测区间）（**仅在线**：[otexts.com/fpp3](https://otexts.com/fpp3/)；2026-10-15 尝试下载整书 PDF 未果，仅拿到 HTML 目录，未入库）状态：待读
- **Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models.** *Biometrika* 65(2):297–303 — 进阶，英文，**Ljung-Box 白噪声检验**的原始论文，本讲（Day46）残差诊断 `ljung_box()` 的直接依据（Oxford，付费）（**仅在线**）状态：待读
- **Chakhchoukh, Y., & Ishikawa, S. *Modeling Data Containing Outliers using ARIMA Additive Outlier (ARIMA-AO)***（arXiv:1803.00257）— 进阶，英文，**已下载** PDF，讲 ARIMA 遇异常点/断点时的加性离群值处理（本讲「停摆阶跃」类问题的进阶解法）→ `Resources/Downloads/Chakhchoukh-Ishikawa-Modeling-Data-Containing-Outliers-using-ARIMA-Additive-Outlier.pdf` 状态：待读
- **Wang, et al. *Why Are the ARIMA and SARIMA not Sufficient***（arXiv:1904.07632）— 进阶，英文，**已下载** PDF，讨论 ARIMA/SARIMA 的适用边界与不足（对本讲「局限」一节的现代视角补充，呼应本讲「强季节要上 SARIMA」）→ `Resources/Downloads/Wang-Why-Are-the-ARIMA-and-SARIMA-not-Sufficient.pdf` 状态：待读
- **Xu, et al. *Forecasting Crime Using ARIMA Model***（arXiv:2003.08006）— 入门/进阶，英文，**已下载** PDF，ARIMA 在城市时序（犯罪量）预测上的完整应用案例，与本讲「景区客流预测」流程同构，可作迁移参考 → `Resources/Downloads/Xu-Forecasting-Crime-Using-ARIMA-Model.pdf` 状态：待读
- **Wikipedia: Autoregressive integrated moving average** — 入门，英文，[Autoregressive integrated moving average](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)（**仅在线**：Wiki 动态页面，无稳定 PDF）状态：待读

> ⚠️ 说明：2026-10-15 备课时宿主机**外网恢复**（arXiv API 与 PDF 下载均可达），故**成功下载 3 篇 arXiv PDF**（ARIMA-AO 1803.00257、Why ARIMA/SARIMA not Sufficient 1904.07632、Forecasting Crime using ARIMA 2003.08006）；Box–Jenkins 1970、Ljung–Box 1978（Wiley/Oxford 付费）与 Hyndman FPP3（在线整书 PDF 下载未果）**仅在线**登记，待机构访问/网络条件允许时补下。

## 灰色系统 / 小样本预测（Level 4）

- **Deng, J. L. (1982). Control problems of grey systems.** *Systems & Control Letters* 1(5):288–294 — 进阶，英文，**灰色系统理论奠基论文**（"灰"= 部分信息已知、部分未知），Day50 GM(1,1) 的思想来源；邓聚龙《灰色系统（社会·经济）》亦为中文原典（Elsevier/华中理工大学出版社，付费）（**仅在线**）状态：待读
- **Deng, J. L. (1989). Introduction to grey system theory.** *The Journal of Grey System* 1(1):1–24 — 入门/进阶，英文，灰色系统理论的系统导读（含 AGO/IAGO、GM(1,1)、后验差检验、灰色关联），Day50 「级比检验 → 建模 → 后验差 C/P」流程的直接出处（JGS，付费）（**仅在线**）状态：待读
- **Liu, S., & Lin, Y. *Grey Systems: Theory and Applications*（Springer, 2010）** — 进阶，英文，灰色系统标准教材，GM(1,1)/DGM(1,1)/Verhulst/滚动 GM 的规范定义与精度等级表，与本讲 `gm11` / `dgm11` 实现逐条对应（Springer，付费）（**仅在线**）状态：待读
- **Ma, X. *A Brief Introduction to the Grey Machine Learning***（arXiv:1805.01745v2）— 进阶，英文，**已下载** PDF，用核方法统一解释灰模型（GM 是"差分 + 核"的特例），Day50 讨论「GM 与线性/指数外推的关系」时的现代视角 → `Resources/Downloads/Ma-A-Brief-Introduction-to-the-Grey-Machine-Learning.pdf` 状态：待读
- **Comert, G., Khan, Z., Rahman, M., & Chowdhury, M. *Grey Models for Short-Term Queue Length Predictions for Adaptive Traffic Signal Control***（arXiv:1912.12676v1）— 进阶，英文，**已下载** PDF，GM(1,1)/Verhulst/滚动 GM 在**只有几个数据点**的实时场景下的工程用法，明确提出"长序列用滚动 GM"，与本讲「末点敏感 + 滚动回测」一节正好互为印证 → `Resources/Downloads/Comert-Grey-Models-for-Short-Term-Queue-Length-Predictions.pdf` 状态：待读
- **Wikipedia: Grey relational analysis / Grey system theory** — 入门，英文，[Grey relational analysis](https://en.wikipedia.org/wiki/Grey_relational_analysis)（**仅在线**：动态页面）状态：待读

> ⚠️ 说明：2026-10-20 备课时宿主机外网**部分可达**（arXiv API/PDF 正常，Springer 正文页返回 HTML 而非 PDF、ScienceDirect 403、Wikipedia 超时），故**成功下载 2 篇 arXiv PDF**（Ma 1805.01745 灰机器学习、Comert 1912.12676 短序列滚动 GM）；Deng 1982/1989 与 Liu–Lin 2010 专著（Elsevier/JGS/Springer 付费）**仅在线**登记。
