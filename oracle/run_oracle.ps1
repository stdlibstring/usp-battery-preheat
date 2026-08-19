# run_oracle.ps1 — Oracle 生成流水线复现脚本
# 用法:
#   .\run_oracle.ps1 -Mode selftest     # 物理内核一致性门禁
#   .\run_oracle.ps1 -Mode dtstudy      # dt 收敛测试(500 随机 case)
#   .\run_oracle.ps1 -Mode generate     # 全量双 Oracle 生成(支持断点续算)
#   .\run_oracle.ps1 -Mode resume       # 从上次中断处继续
param(
    [Parameter(Position = 0)]
    [ValidateSet("selftest", "dtstudy", "generate", "resume")]
    [string]$Mode = "selftest",

    [double]$Dt = 0.5,              # 全量生成步长(收敛测试结论:0.5 已足够)
    [long]$Audit = 200,             # 0.01s 审计 case 数
    [string]$Input = "..\sdk\data\nav_train_100000.txt",
    [string]$OutPrefix = "output/oracle"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 编译(与提交代码相同严格等级)
gcc -Wall -Wextra -Wpedantic -O2 -std=c11 -fopenmp `
    -o oracle_generator.exe oracle_generator.c -lm
if ($LASTEXITCODE -ne 0) { throw "compile failed" }

switch ($Mode) {
    "selftest" { .\oracle_generator.exe --selftest }
    "dtstudy"  { .\oracle_generator.exe --dt-study --n 500 --seed 42 --input $Input }
    "generate" {
        # 门禁必须先通过
        .\oracle_generator.exe --selftest
        if ($LASTEXITCODE -ne 0) { throw "selftest gate FAILED - refusing to generate" }
        .\oracle_generator.exe --generate --dt $Dt --audit $Audit `
            --input $Input --out-prefix $OutPrefix
    }
    "resume"   {
        .\oracle_generator.exe --generate --dt $Dt --audit 0 --resume `
            --input $Input --out-prefix $OutPrefix
    }
}
exit $LASTEXITCODE
