# Level1_EncryptDigits

对应 [Learning_Roadmap.md](../../Learning_Roadmap.md) 的 **Level 1：类型与基础语法**。

## 目的

练习十进制取位、取余运算、`std::swap` 与输入输出；同时修正作业中"千位取成 `x%1000`"的 bug。

## 规则

对一个 4 位整数，记其各位为 a、b、c、d（千、百、十、个）：

1. 每位数字加 7 后对 10 取余；
2. 交换第 1 位与第 3 位、第 2 位与第 4 位；
3. 输出加密后的四位数。

## 文件

- `encrypt.cpp` —— 唯一源文件（含 `main`，从标准输入读两个 4 位数）。

## 编译与运行

```powershell
cd "D:\Run_Tao\Note\C++\Code\Level1_EncryptDigits"
g++ -std=c++20 -Wall -Wextra -O2 encrypt.cpp -o encrypt.exe
# 输入示例：
# 1234 5678
.\encrypt.exe
```

## 预期输出

输入 `1234 5678`：

```
189 543
```

（1234 加密后为 0189，按整数输出即 189；5678 加密后为 0543，按整数输出即 543。）

## 手算校验

以 1234 为例：各位加 7 取余得 8、9、0、1；交换第 1/3 位、第 2/4 位得 0、1、8、9，即 0189。

> [!warning] 原作业的 bug
> 原代码用 `step_1(x%1000)` 取千位，得到的是后三位，结果为 119 而非 189。
> 正确写法见 `encrypt.cpp` 中的 `(x / 1000) % 10`。