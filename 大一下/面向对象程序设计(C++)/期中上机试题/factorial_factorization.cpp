#include <iostream>

using namespace std;

// 数据范围 N <= 10^6
const int MAXN = 1000005;
int primes[MAXN], cnt = 0;
bool is_not_prime[MAXN];

/**
 * 线性筛法 (欧拉筛)
 * 在 O(N) 时间复杂度内找出 1 到 n 之间的所有质数
 */
void sieve(int n) {
    is_not_prime[0] = is_not_prime[1] = true;
    for (int i = 2; i <= n; i++) {
        if (!is_not_prime[i]) {
            primes[cnt++] = i;
        }
        for (int j = 0; j < cnt && i * primes[j] <= n; j++) {
            is_not_prime[i * primes[j]] = true;
            if (i % primes[j] == 0) break; // 线性筛的核心：每个合数只被其最小质因子筛去
        }
    }
}

/**
 * 计算 N! 分解质因数
 * 对于每个质数 p，其在 N! 中的幂次 c = floor(N/p) + floor(N/p^2) + floor(N/p^3) + ...
 */
int main() {
    // 优化 I/O 速度
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    int n;
    if (!(cin >> n)) return 0;
    
    // 1. 预处理出所有小于等于 n 的质数
    sieve(n);
    
    // 2. 依次计算每个质数 p 的幂次并输出
    for (int i = 0; i < cnt; i++) {
        int p = primes[i];
        long long exponent = 0;
        int temp_n = n;
        
        // Legendre's Formula 实现
        while (temp_n >= p) {
            exponent += (temp_n / p);
            temp_n /= p;
        }
        
        cout << p << " " << exponent << "\n";
    }
    
    return 0;
}
