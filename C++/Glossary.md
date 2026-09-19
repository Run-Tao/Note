---
title: C++ Glossary
type: index
subject: cpp
level: overview
tags:
  - programming
  - cpp
  - glossary
---

# Glossary 术语表

> C++ 重要术语中英文对照。按主题分组，随学习进度补充。
> 术语定义详见对应笔记，这里只给一句话定位。

## 语言与编译（Level 0–1）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 翻译单元 | translation unit | 一个源文件经预处理后的完整编译输入 | [[Level0_Environment_and_Compilation]] |
| 预处理 | preprocessing | 处理 `#include` / `#define` 等指令的阶段 | [[Level0_Environment_and_Compilation]] |
| 编译 / 汇编 / 链接 | compile / assemble / link | 源码 → 汇编 → 目标文件 → 可执行文件 | [[Level0_Environment_and_Compilation]] |
| 头文件 / 源文件 | header / source file | 声明放头文件，实现放源文件 | [[Level0_Environment_and_Compilation]] |
| 声明 / 定义 | declaration / definition | 告诉编译器存在 / 真正分配实体 | [[Level0_Environment_and_Compilation]] |
| 未定义行为 | undefined behavior (UB) | 标准未规定结果，任何事都可能发生 | [[Level4_Memory_and_Pointers]] |
| 冯诺依曼架构 | von Neumann architecture | 存储程序式计算机：输入 / 输出 / 存储器 / 运算器 / 控制器 | [[Level0_Environment_and_Compilation]] |
| 流程图 | flowchart | 用图形符号描述程序控制流 | [[Level0_Environment_and_Compilation]] |
| 盒图 / N-S 图 | Nassi–Shneiderman diagram | 用嵌套矩形表示顺序 / 选择 / 循环 | [[Level0_Environment_and_Compilation]] |
| cout / cin | standard output / input stream | 标准输出 / 输入流，定义于 `<iostream>` | [[Level0_Environment_and_Compilation]] |
| endl / flush / ends | stream manipulators | 分别插入换行 / 无 / `'\0'`，并刷新缓冲区 | [[Level0_Environment_and_Compilation]] |
| 缓冲区 | buffer | 内存中的临时存储，刷新时写出 | [[Level0_Environment_and_Compilation]] |

## 类型与表达式（Level 1–2）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 左值 / 右值 | lvalue / rvalue | 有身份可寻址 / 通常是临时值 | [[Level7_Modern_Cpp_Idioms]] |
| 值类别 | value category | 表达式的左值 / 右值分类 | [[Level7_Modern_Cpp_Idioms]] |
| 常量表达式 | constexpr | 编译期即可求值的表达式 | [[Level5_Templates_and_Generics]] |
| 类型推导 | type deduction | `auto` / 模板自动确定类型 | [[Level5_Templates_and_Generics]] |
| 隐式转换 | implicit conversion | 编译器自动做的类型转换 | [[Level1_Types_and_Syntax]] |
| 变量 | variable | 内存中被命名的存储空间 | [[Level1_Types_and_Syntax]] |
| 数据类型 | data type | 决定大小 / 布局 / 取值范围 / 可参与的运算 | [[Level1_Types_and_Syntax]] |
| 标识符 | identifier | 变量等的名字；字母 / 下划线开头，区分大小写 | [[Level1_Types_and_Syntax]] |
| 直接初始化 / 列表初始化 | direct / list initialization | `int a(2)` / `int a{2}`；列表初始化禁止窄化转换 | [[Level1_Types_and_Syntax]] |
| 原码 / 反码 / 补码 | sign-magnitude / ones' / two's complement | 整数的机器编码；现代计算机用补码 | [[Level1_Types_and_Syntax]] |
| 枚举 / 枚举类 | enum / enum class | 一组具名整型常量；`enum class` 作用域受限 | [[Level1_Types_and_Syntax]] |
| 引用 | reference | 变量的别名，必须绑定且不可重绑 | [[Level2_Functions_and_Scope]] |
| 命名空间 | namespace | 避免名字冲突的作用域划分 | [[Level2_Functions_and_Scope]] |

## 面向对象与资源管理（Level 3–4）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 类 / 对象 | class / object | 类型定义 / 该类型的实例 | [[Level3_Object_Oriented]] |
| 构造 / 析构函数 | constructor / destructor | 对象建立 / 销毁时自动调用 | [[Level3_Object_Oriented]] |
| RAII | Resource Acquisition Is Initialization | 用对象生命周期管理资源 | [[Level3_Object_Oriented]] |
| 拷贝控制 | copy control | 拷贝构造、拷贝赋值、析构等 | [[Level3_Object_Oriented]] |
| 多态 / 虚函数 | polymorphism / virtual function | 运行时按实际类型调用 | [[Level3_Object_Oriented]] |
| 栈 / 堆 | stack / heap | 自动管理 / 手动或智能指针管理 | [[Level4_Memory_and_Pointers]] |
| 智能指针 | smart pointer | `unique_ptr` / `shared_ptr` / `weak_ptr` | [[Level4_Memory_and_Pointers]] |
| 悬垂指针 | dangling pointer | 指向已释放内存的指针 | [[Level4_Memory_and_Pointers]] |
| 所有权 | ownership | 谁负责释放资源的语义 | [[Level4_Memory_and_Pointers]] |

## 泛型、STL 与现代惯用法（Level 5–7）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 模板 | template | 参数化的类型 / 值的代码 | [[Level5_Templates_and_Generics]] |
| 实例化 | instantiation | 用具体类型生成代码 | [[Level5_Templates_and_Generics]] |
| 概念 / 约束 | concept / constraint | C++20 对模板参数的要求 | [[Level5_Templates_and_Generics]] |
| 容器 | container | `vector` / `map` 等数据结构 | [[Level6_STL_and_Algorithms]] |
| 迭代器 | iterator | 遍历容器的统一接口 | [[Level6_STL_and_Algorithms]] |
| 算法 | algorithm | `<algorithm>` 中的泛型操作 | [[Level6_STL_and_Algorithms]] |
| lambda 表达式 | lambda expression | 匿名的函数对象 | [[Level6_STL_and_Algorithms]] |
| 移动语义 | move semantics | 转移资源而非拷贝 | [[Level7_Modern_Cpp_Idioms]] |
| 完美转发 | perfect forwarding | 保持值类别的转发 | [[Level7_Modern_Cpp_Idioms]] |
| 异常安全 | exception safety | 异常发生时仍保持一致状态 | [[Level7_Modern_Cpp_Idioms]] |
| 视图 | string_view / span / view | 不拥有数据的轻量引用 | [[Level7_Modern_Cpp_Idioms]] |

## 并发与工程（Level 8–9）

| 中文 | English | 一句话 | 相关笔记 |
| --- | --- | --- | --- |
| 线程 | thread | 程序内的独立执行流 | [[Level8_Concurrency_and_Performance]] |
| 互斥量 | mutex | 保护共享数据的锁 | [[Level8_Concurrency_and_Performance]] |
| 条件变量 | condition variable | 线程间等待 / 通知 | [[Level8_Concurrency_and_Performance]] |
| 原子操作 | atomic | 不可分割的操作 | [[Level8_Concurrency_and_Performance]] |
| 数据竞争 | data race | 并发访问同一数据且有写 | [[Level8_Concurrency_and_Performance]] |
| 内存序 | memory order | 原子操作间的可见性约定 | [[Level8_Concurrency_and_Performance]] |
| 构建系统 | build system | CMake 等组织编译的配置 | [[Level9_Engineering_and_Practice]] |
| 静态分析 / 消毒器 | static analysis / sanitizer | 发现潜在缺陷的工具 | [[Level9_Engineering_and_Practice]] |
