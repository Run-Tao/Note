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

