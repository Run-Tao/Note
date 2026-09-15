---
title: C++ Learning Roadmap
type: roadmap
subject: cpp
level: overview
tags:
  - programming
  - cpp
  - roadmap
---

# C++ 学习路线（Level 0 – Level 9）

> 面向人工智能 / 计算机方向本科生：从零系统学习现代 C++，一路通向并发、性能与工程实践。
>
> 原则：**理解语言模型优先于背语法，写正确代码优先于写炫技代码，应用优先于纸面知识。**

## 路线总览

| Level | 主题 | 前置知识 | 一句话目标 |
| --- | --- | --- | --- |
| Level 0 | 环境与编译 | 无 | 打通"写 → 编译 → 运行 → 调试"链路 |
| Level 1 | 类型与基础语法 | Level 0 | 理解类型系统与表达式求值 |
| Level 2 | 函数与作用域 | Level 1 | 掌握函数、重载、引用与生命周期 |
| Level 3 | 面向对象 | Level 2 | 用类与 RAII 组织代码 |
| Level 4 | 内存与指针 | Level 2–3 | 搞清栈 / 堆、指针与所有权 |
| Level 5 | 模板与泛型 | Level 2–4 | 用模板写泛型代码，理解编译期计算 |
| Level 6 | STL 与算法 | Level 2–5 | 熟练使用容器、迭代器与算法库 |
| Level 7 | 现代 C++ 惯用法 | Level 3–6 | move 语义、异常安全、Concepts / Ranges |
| Level 8 | 并发与性能 | Level 4–7 | 线程、同步、内存模型与优化 |
| Level 9 | 工程与实战 | Level 0–8 | 构建、测试、调试，面向算法与 AI 工程 |

## 使用说明

- 按 Level 顺序学习；有基础者可跳过对应 Level，但建议至少浏览一遍；
- 每个 Level 的完成标准：能用一两句话讲清核心概念，写出可编译运行且无警告的代码，并在真实场景中找到至少一个例子；
- 学完一个 Level 后更新文末进度表，并同步在 Notes / Exercises / Code 中留下记录；遗留疑问用 `> [!question]` 标记。

---

## Level 0：环境与编译

**学习目标**

- 能安装并使用一个 C++ 工具链，理解"源码 → 目标文件 → 可链接可执行文件"的过程；
- 理解编译错误 / 链接错误 / 运行错误三者的区别；
- 掌握基本的调试手段（打印、断点、编译器警告）。

**核心概念**

- 编译器、链接器、翻译单元、头文件与源文件；
- 预处理、编译、汇编、链接四个阶段；
- `main` 函数、标准输入输出 `iostream`；
- 编译选项：`-std=`、`-Wall -Wextra`、`-O2`、`-g`。

**前置知识**

- 无（具备基本编程概念即可）。

**应用**

- 任何 C++ 项目的起点；理解构建流程是排查一切构建问题的前提；
- 算法竞赛 / 刷题环境的快速搭建。

**推荐资料**

- cppreference（编译器与标准库权威参考）；
- 《C++ Primer》第 1–2 章；
- learncpp.com 第 0 章。

## Level 1：类型与基础语法

**学习目标**

- 理解 C++ 是静态类型语言，掌握基本类型与类型转换；
- 熟练使用变量、常量、运算符、控制流；
- 理解表达式的值类别（左值 / 右值）初识。

**核心概念**

- 基本类型（`int / double / char / bool`）、`auto`、固定宽度整数；
- 变量初始化、`const` / `constexpr`；
- 运算符与优先级、隐式与显式类型转换；
- 条件、循环、`switch`；控制流与作用域初识。

**前置知识**

- Level 0。

**应用**

- 一切代码的基础；正确的类型与初始化是避免 UB 的第一步。

**推荐资料**

- 《C++ Primer》第 2–5 章；
- learncpp.com 第 1–7 章。

## Level 2：函数与作用域

**学习目标**

- 掌握函数定义、声明、重载、默认参数；
- 理解值传递、引用传递与 `const` 引用；
- 理解作用域、生命周期与名字查找。

**核心概念**

- 函数、参数传递、返回值、函数重载、默认参数；
- 引用与 `const` 引用、左值 / 右值初识；
- 作用域、命名空间、内联函数、函数指针；
- 递归与栈帧直觉。

**前置知识**

- Level 1。

**应用**

- 模块化组织代码；算法实现中函数设计的基础。

**推荐资料**

- 《C++ Primer》第 6 章；
- learncpp.com 第 2、8 章。

## Level 3：面向对象

**学习目标**

- 用类与对象封装数据与行为；
- 理解构造 / 析构 / 拷贝控制与 RAII 思想；
- 理解继承、多态与虚函数（以及代价）。

**核心概念**

- 类、成员、访问控制、`this`；
- 构造函数、析构函数、初始化列表；
- 拷贝构造 / 拷贝赋值、`= default` / `= delete`；
- 继承、虚函数、多态、抽象类、`override` / `final`；
- 运算符重载、友元。

**前置知识**

- Level 2。

**应用**

- 用 RAII 管理资源（内存、文件、锁）；设计可扩展的类型体系。

**推荐资料**

- 《C++ Primer》第 7、13、15 章；
- Effective C++ / Effective Modern C++ 相关条款。

## Level 4：内存与指针

**学习目标**

- 清晰区分栈与堆、值语义与引用语义；
- 掌握指针、引用、数组与动态内存；
- 理解智能指针与所有权的现代做法。

**核心概念**

- 栈 / 堆 / 静态存储区、内存布局；
- 指针、指针运算、`nullptr`、数组与指针关系；
- `new` / `delete`（理解但尽量不用裸的）；
- 智能指针 `unique_ptr` / `shared_ptr` / `weak_ptr`；
- 悬垂指针、内存泄漏、未定义行为。

**前置知识**

- Level 2–3。

**应用**

- 所有需要手动管理资源或与 C 接口交互的场景；
- 数据结构的动态内存实现（链表、树、图）。

**推荐资料**

- 《C++ Primer》第 12 章；
- Effective Modern C++ 第 4 章（智能指针）。

## Level 5：模板与泛型

**学习目标**

- 用函数模板与类模板编写泛型代码；
- 理解模板实例化的机制与编译期计算；
- 初步接触 `constexpr`、可变参数模板与 Concepts。

**核心概念**

- 函数模板、类模板、模板参数推导；
- 特化与偏特化、SFINAE 直觉；
- `constexpr` / 编译期计算；
- 可变参数模板、折叠表达式；
- Concepts（C++20）与约束。

**前置知识**

- Level 2–4。

**应用**

- STL 的底层实现；编写可复用的库代码；高性能数值计算。

**推荐资料**

- 《C++ Primer》第 16 章；
- Effective Modern C++ 第 1 章（类型推导）；
- cppreference：Concepts。

## Level 6：STL 与算法

**学习目标**

- 熟练选用容器，理解各自复杂度与适用场景；
- 掌握迭代器与算法库，写出惯用的 STL 代码；
- 理解 lambda、函数对象与算法配合。

**核心概念**

- 序列容器：`vector` / `deque` / `list` / `array`；
- 关联容器：`map` / `set` / `unordered_map` / `unordered_set`；
- 容器适配器：`stack` / `queue` / `priority_queue`；
- 迭代器类别、范围 `for`、算法库（`<algorithm>`）；
- lambda 表达式与捕获。

**前置知识**

- Level 2–5。

**应用**

- 算法题、数据处理、日常工程；STL 是 C++ 生产力的核心。

**推荐资料**

- 《C++ Primer》第 9–11、14 章；
- cppreference：Containers / Algorithms。

## Level 7：现代 C++ 惯用法

**学习目标**

- 理解移动语义与完美转发，写出高效的资源管理代码；
- 掌握异常安全、RAII、`Rule of Zero` 等惯用法；
- 掌握 Concepts / Ranges 等现代设施。

**核心概念**

- 左值 / 右值 / 值类别、`std::move`、`std::forward`、移动构造 / 赋值；
- 拷贝省略、返回值优化（RVO / NRVO）；
- 异常与异常安全保证；
- Rule of Three / Five / Zero；
- `std::optional` / `std::variant` / `std::string_view`；
- Ranges（C++20）、结构化绑定。

**前置知识**

- Level 3–6。

**应用**

- 高性能与可维护的现代 C++ 代码；库接口设计。

**推荐资料**

- Effective Modern C++；
- 《Effective C++》与 Meyers 的现代 C++ 系列；
- cppreference：Move semantics、Ranges。

## Level 8：并发与性能

**学习目标**

- 掌握线程、互斥、条件变量与原子操作；
- 理解 C++ 内存模型与数据竞争；
- 掌握性能分析方法，能定位并优化热点。

**核心概念**

- `std::thread`、`jthread`、`std::mutex`、锁与 RAII；
- 条件变量、生产者-消费者模型；
- `std::atomic`、内存序（memory order）；
- 线程池、任务并行、异步 `std::async` / `future`；
- cache、分支预测、profiling 与基准测试。

**前置知识**

- Level 4–7。

**应用**

- 高性能计算、服务器、并行算法；AI 推理 / 训练中的并行。

**推荐资料**

- 《C++ Concurrency in Action》（Anthony Williams）；
- 《计算机组成》与性能优化相关资料；
- Google Benchmark、perf / VTune 等工具文档。

## Level 9：工程与实战

**学习目标**

- 会用 CMake 组织多文件项目，管理依赖与构建配置；
- 掌握单元测试与调试工具，能定位并修复缺陷；
- 了解 C++ 在算法竞赛与 AI 工程中的实际工程形态。

**核心概念**

- CMake / 构建系统、头文件组织、库与可执行目标；
- 测试框架（Catch2 / GoogleTest / doctest）与 TDD 思路；
- 调试器（GDB / LLDB / VS）、sanitizers（ASan / UBSan / TSan）；
- 静态分析（clang-tidy / cppcheck）、格式化（clang-format）；
- 包管理（vcpkg / Conan）与版本控制协作。

**前置知识**

- Level 0–8。

**应用**

- 真实项目的构建与质量保障；为算法、系统、AI 工程打下工程基础。

**推荐资料**

- CMake 官方教程 / 《CMake 实践》；
- GoogleTest、Catch2 官方文档；
- Effective C++ 系列与工程实践博客。

---

## 进度追踪

| Level | 主题 | 状态（未开始 / 学习中 / 完成） | 笔记 | 练习 |
| --- | --- | --- | --- | --- |
| Level 0 | 环境与编译 | 未开始 | | |
| Level 1 | 类型与基础语法 | 未开始 | | |
| Level 2 | 函数与作用域 | 未开始 | | |
| Level 3 | 面向对象 | 未开始 | | |
| Level 4 | 内存与指针 | 未开始 | | |
| Level 5 | 模板与泛型 | 未开始 | | |
| Level 6 | STL 与算法 | 未开始 | | |
| Level 7 | 现代 C++ 惯用法 | 未开始 | | |
| Level 8 | 并发与性能 | 未开始 | | |
| Level 9 | 工程与实战 | 未开始 | | |

> 提示：完成一个 Level 后，在对应目录创建笔记与练习文件，并把链接填进上表。
