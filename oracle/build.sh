#!/bin/bash
# build.sh — Linux 服务器上构建 Oracle 生成器与物理动态库
# 用法: cd oracle && bash build.sh
set -e
cd "$(dirname "$0")"

# 需要 gcc 支持 C11 + OpenMP
gcc -Wall -Wextra -Wpedantic -O2 -std=c11 -fopenmp \
    -o oracle_generator oracle_generator.c -lm
echo "[ok] oracle_generator"

gcc -Wall -Wextra -O2 -std=c11 -shared -fPIC \
    -o libphysics.so physics.c -lm
echo "[ok] libphysics.so"

# 快速验证: 物理内核必须复现 Mock 基准
./oracle_generator --selftest
