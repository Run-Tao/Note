# Level0_HelloWorld

对应 [Learning_Roadmap.md](../../Learning_Roadmap.md) 的 **Level 0：环境与编译**。

## 目的

验证 C++ 工具链与 VS Code 配置是否可用，同时观察编译器版本、语言标准与基本类型大小。

## 文件

- `hello.cpp` —— 环境自检程序（唯一源文件）。

## 编译与运行

### 方式 1：VS Code（推荐）

1. 用 VS Code 打开笔记库根目录 `D:\Run_Tao\Note`；
2. 打开 `hello.cpp`；
3. 按 `Ctrl+Shift+B` 生成（默认任务：`C++: 生成活动文件 (Debug)`）；
4. 按 `F5` 调试运行，或 `Ctrl+F5` 直接运行。

其他任务（`Ctrl+Shift+P` → `Tasks: Run Task`）：

| 任务 | 作用 |
| --- | --- |
| `C++: 生成活动文件 (Debug)` | `-g` + 全部警告，用于开发与调试（默认） |
| `C++: 生成活动文件 (Release)` | `-O2` 优化构建，用于性能测试 |
| `C++: 生成并运行活动文件` | 编译后立即运行 |
| `C++: 用 g++ 多文件编译` | 编译当前目录下所有 `.cpp` |

### 方式 2：终端

```powershell
cd "D:\Run_Tao\Note\C++\Code\Level0_HelloWorld"
g++ -std=c++17 -Wall -Wextra -g hello.cpp -o hello.exe
.\hello.exe
```

### 方式 3：Code Runner

装好 Code Runner 扩展后，右键编辑器 → **Run Code**（或 `Ctrl+Alt+N`）。

## 预期输出

```
=== C++ 环境自检 ===
C++ 标准宏 __cplusplus = 201703
编译器: GCC 8.1.0
sizeof(int)    = 4
sizeof(long)   = 4
sizeof(void*)  = 8（64 位平台应为 8）
排序后的语言: [C++] [Python] [Rust]
=== 环境正常，可以开始学习 ===
```

> [!warning] 注意
> `sizeof(long)` 在 64 位 Windows 上是 **4**（LLP64 模型），而在 64 位 Linux 上是 8。这是平台差异的经典例子，写跨平台代码时应使用 `<cstdint>` 的 `int32_t` / `int64_t`。

## 动手改一改

1. 把 `-std=c++17` 改成 `-std=c++11`，观察 `__cplusplus` 的变化；
2. 故意删掉一个 `;`，看编译器报错信息如何定位；
3. 用 `g++ -E hello.cpp -o hello.i` 查看预处理结果，体会 `#include` 到底做了什么。
