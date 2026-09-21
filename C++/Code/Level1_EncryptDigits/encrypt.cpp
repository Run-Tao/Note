// 四位数加密（修正版）
// 规则：每位数字加 7 后除以 10 取余；再交换第 1、3 位与第 2、4 位。
// 编译：g++ -std=c++20 -Wall -Wextra -O2 encrypt.cpp -o encrypt

#include <cassert>
#include <iostream>
#include <utility>

namespace {
constexpr int kAddend = 7;
constexpr int kBase = 10;

// 对单个十进制数字加密：加 7 后对 10 取余
int addSevenModTen(int digit) {
    return (digit + kAddend) % kBase;
}
}  // namespace

// 输入必须是 4 位整数（1000 - 9999）
int encrypt(int x) {
    assert(x >= 1000 && x <= 9999);

    int a = addSevenModTen((x / 1000) % 10);  // 千位
    int b = addSevenModTen((x / 100) % 10);   // 百位
    int c = addSevenModTen((x / 10) % 10);    // 十位
    int d = addSevenModTen(x % 10);           // 个位

    std::swap(a, c);  // 交换第 1、3 位
    std::swap(b, d);  // 交换第 2、4 位

    return a * 1000 + b * 100 + c * 10 + d;
}

int main() {
    int m = 0;
    int n = 0;
    std::cin >> m >> n;
    std::cout << encrypt(m) << ' ' << encrypt(n) << '\n';
    return 0;
}