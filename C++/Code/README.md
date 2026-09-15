# Code 代码

可编译运行的 C++ 源码，按 Level 组织，与 `Notes/`、`Exercises/`、`Projects/` 配合使用。

## 目录约定

```
Code/
├── Level0_HelloWorld/
├── ...
└── README.md（本文件）
```

每个子目录是一个独立可编译单元，建议包含：

- 源码（`.cpp` / `.h`）；
- `README.md`：说明做什么、怎么编译、预期输出；
- （可选）`CMakeLists.txt`。

## 编译约定

推荐使用现代 C++ 标准与警告选项：

```bash
# g++ / clang++
g++ -std=c++20 -Wall -Wextra -Wpedantic -O2 -g main.cpp -o main

# MSVC
cl /std:c++20 /W4 /O2 main.cpp
```

涉及性能的结论请标注编译器、版本与优化选项。
