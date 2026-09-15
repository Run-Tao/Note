// hello.cpp —— 环境自检示例
//
// 用途：验证 C++ 工具链与 VS Code 配置是否正常。
// 编译（终端）：g++ -std=c++17 -Wall -Wextra -g hello.cpp -o hello.exe
// 运行（终端）：.\hello.exe
// VS Code：打开本文件 → Ctrl+Shift+B 生成 → F5 调试

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "=== C++ 环境自检 ===\n";

    // 1. 语言标准（201703 表示 C++17）
    std::cout << "C++ 标准宏 __cplusplus = " << __cplusplus << "\n";

    // 2. 编译器标识
#if defined(__clang__)
    std::cout << "编译器: Clang " << __clang_major__ << "." << __clang_minor__ << "\n";
#elif defined(__GNUC__)
    std::cout << "编译器: GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__ << "\n";
#elif defined(_MSC_VER)
    std::cout << "编译器: MSVC " << _MSC_VER << "\n";
#else
    std::cout << "编译器: 未知\n";
#endif

    // 3. 基本类型大小（理解平台差异的起点）
    std::cout << "sizeof(int)    = " << sizeof(int) << "\n";
    std::cout << "sizeof(long)   = " << sizeof(long) << "\n";
    std::cout << "sizeof(void*)  = " << sizeof(void*) << "（64 位平台应为 8）\n";

    // 4. 跑一段真实的 C++17 代码：vector + 算法 + 范围 for
    std::vector<std::string> lang{"C++", "Python", "Rust"};
    std::sort(lang.begin(), lang.end());
    std::cout << "排序后的语言:";
    for (const auto& s : lang) std::cout << " [" << s << "]";
    std::cout << "\n";

    std::cout << "=== 环境正常，可以开始学习 ===\n";
    return 0;
}
