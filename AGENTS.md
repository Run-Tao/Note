# AGENTS.md — 项目执行规范

> 本文件是 `D:\Run_Tao\Note`（Obsidian 数学学习知识库）的**项目执行规范**，供 Agent 执行任务时遵守。
> 它是执行规则，不是数学知识本身；知识内容与写作规范以各知识库的 config / README 为准。

## 1. 仓库组织方式

- 根目录是一个 Obsidian Vault，含**两个独立知识库** + **个人层** + `.obsidian/` 配置；
- 每个知识库 = README（介绍）+ Learning_Roadmap（学习路线）+ 00 - MOC/（总览页）+ 按主题编号的子目录（Notes / Exercises / Projects / Resources 等）；
- 每个知识库含 `Inbox/`：原始笔记（每天一个文件 `YYYY-MM-DD.md`，内容随意碎片化，永不删除）；Agent 提取内容归入正式笔记后，整个文件移入 `Inbox/Processed/` 归档；
- 知识工作流（学习 → Inbox → Agent 整理 → 正式知识库）见根目录 [[Knowledge_Workflow]]；
- 两个知识库之间通过 Wiki Links 互相引用；
- **个人层**（`00 - Personal/` + 根目录计划文档）是持续性个人知识库的骨架：
  - `00 - Personal/Personal Hub.md`：个人总览页，串联学科知识库、计划、学习日志；
  - `00 - Personal/Learning Log/`：学习日志（每周一篇 + 月复盘），让知识库随时间持续生长；
  - 根目录计划文档（如 `Summer_Plan.md`）：按学期更新的计划，执行完保留为历史存档；
- 维护节奏：每周学习日志 → 每学期更新计划与能力地图；新计划 / 课程笔记按需创建，不批量预建空文件。

## 2. 两个知识库的用途

| 知识库 | 用途 | 现状 |
| --- | --- | --- |
| `Linear_Algebra/` | 面向 AI/计算机本科生的线性代数系统学习（Level 0–8，概念 → AI 应用）；规范权威是 `config.md` | 框架完整，Notes 已完成 Level 0–2 |
| `Mathematical_Modeling/` | 数学建模竞赛能力体系（Level 0–12：现实问题 → 建模 → 算法 → Python → 验证 → 论文） | 仅 README / 路线 / MOC 骨架，各子目录为空 |
| `00 - Personal/`（个人层） | 持续性个人知识库骨架：个人总览、学习日志、按学期更新的计划文档 | 已建立（2026-08）：Hub + Learning Log |

- 执行任务前先读目标知识库的 README 与 config.md（如有），以其为准。
- **学习者背景**：用户为准大一新生，**高中数学已扎实**（函数、三角、解析几何、基础代数）；涉及"高中知识"时不复习、不铺垫、不解释基础概念，直接进入大学数学内容。

## 3. Markdown / LaTeX / Wiki Links 使用方式

- 中文 Markdown；公式用 Obsidian 原生 LaTeX：行内 `$...$`，独立公式 `$$...$$`；
- Wiki Links：`[[文件名]]` 或 `[[文件名|显示名]]`；优先引用已有文件；未存在的概念可建链接，但不为建链而创建空白文件；
- YAML frontmatter：`title / type / subject / level / tags`（tags 精简，不堆砌）；
- Callout：适度使用 `> [!note]` / `> [!tip]` / `> [!example]` / `> [!warning]` / `> [!abstract]`；
- 知识笔记遵循各库模板（直觉 → 定义 → 公式 → 常见误解 → 应用 → 遗留疑问）。

## 4. 修改文件的原则

- **增量修改**：不删除正确内容、不无理由重写、不改变已有知识结构；
- 数学错误可修正，但保留结构、链接与 Wiki Links；
- 重命名 / 移动文件前，先检查并更新所有引用它的 Wiki Links；
- 修改 config.md 等规范文件属于重大变更：先说明理由，再同步更新 README 与相关模板。

## 5. 不重复已有知识

- 动手前先搜索 / 阅读相关笔记，确认内容是否已存在；
- 已有内容直接引用（Wiki Link），不重复写入；
- 新内容只写"已有笔记之外"的部分，避免产生内容重叠的文件。

## 6. 不随意修改无关文件

- 每次任务只触碰与任务直接相关的文件；
- 不清理、不重排、不"顺手优化"无关目录（含 `00 - Personal/组会旁听.md` 等非任务文件）；
- 归属不明确的内容先记录到 Inbox/ 或询问，不擅自移动或删除。

## 6.5 参考阅读资料的下载

- 生成参考阅读资料（论文、官方文档、PDF、网页等）时，凡可下载的文档一律**下载到对应知识库的 `Resources/Downloads/`**，不只在笔记里留链接；
- 下载规则：文件名清晰（`作者-标题.pdf` 或 `标题.pdf`），避免无意义文件名；网页可保存为 Markdown / PDF；
- 下载后：① 在对应 `Resources/README.md` 的索引中登记（名称、难度、语言、推荐理由、本地路径）；② 在笔记中用 Wiki Link 或相对路径引用本地文件，而不是外链；
- 无法下载 / 仅限在线浏览的（如视频、需登录的资源）才只记录链接，并标注"仅在线"；
- 下载位置只限各知识库 `Resources/Downloads/`，不散落在其他目录。

## 7. 数学知识 ↔ 建模 ↔ Python 的链接

- **跨库链接**：建模库引用线代库的既有笔记（如 `[[Level1_Vectors_and_Matrices]]`）；线代库的 AI 应用笔记链接到建模 / ML 相关内容；
- **Python 项目**：`Projects/` 存放"数学概念 → Python 实验 → 可视化 / 应用 → 项目记录"；建模库 `08 - Python/` 汇总建模所用库（NumPy / Pandas / SciPy / scikit-learn 等）；
- 知识笔记中只记录代码位置、关键结果与验证结论，不把大段完整代码塞进笔记；完整代码放 Projects/ 或代码文件。

## 8. 输出长度控制

- 优先完成文件修改，而非长篇聊天解释；
- 回复简洁：说清"做了什么、改了哪些文件、下一步是什么"即可；
- 写入文件的内容同样克制：只写必要信息，不堆砌已有内容；
- 需要长内容时写入文件，不在对话里展开。

## 9. 执行环境与常见报错规避（2026-08-22 增补）

本会话在 DSH 沙箱中运行，以下规则避免重复踩坑：

- **编辑/覆盖文件前必须先 read**（含覆盖 write）；同一文件的多个修改**串行**执行，勿并发编辑同一文件（会报 `ReplaceFileW EIO`）；
- **建目录**：不要用 PowerShell `New-Item`（沙箱拒绝），改用 write 工具写入占位文件（`.gitkeep` / `README.md`）隐式创建目录；
- **git 全局配置**被沙箱锁定（`--global` 报 Permission denied），一律用 `--local`（仓库级 `.git/config`）；
- **网络**：直连 GitHub 不通，必须走本地代理；git 代理与 `http.sslBackend=openssl` 已配置于 `.git/config`（沙箱 schannel 证书不可用，勿改回）；不要用 `Test-NetConnection`（慢且易超时），用 curl 快速探测；
- **认证**：沙箱禁交互进程，git 登录弹窗/askpass 不可用（`sh.exe signal pipe` 报错）；用便携版 gh CLI（`%LOCALAPPDATA%\gh-cli\bin\gh.exe`）设备码流程 `gh auth login --web`；
- **git 退出码误报**：PowerShell 将 git 的 stderr 输出当错误显示且 exit code=1，但**实际可能成功**；判断成功看输出关键词（`[new branch]`、`main -> main`、`Authentication complete`），而非 exit code。
