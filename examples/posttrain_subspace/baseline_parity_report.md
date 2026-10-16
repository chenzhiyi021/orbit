# Baseline parity report

本文件是代码迁移验收记录，不是正式训练结果。机器可读结果见 [baseline_parity_report.json](baseline_parity_report.json)。

CPU检查在隔离的 Python3.12.3 / PyTorch2.11.0+cpu / NumPy1.26.4 环境执行。生产数学函数与封存M6原函数体共同参与测试；去掉fixture的torch.compile装饰器以避免把编译启动混入小词表测试，函数体保持原样。

| 项目 | 当前证据 |
|---|---|
| M6固定prefix枚举 | 64种词表/温度/seed/scale；新生产路径与legacy helper梯度逐项相同；与full RKL在声明的FP32量级容差内；p=q为零；动作无关baseline不改期望梯度 |
| 原LR trace | 100个实际applied LR逐项精确匹配；第1步0、第11步峰值；未增加第101次更新 |
| 原初始化 | name-seeded FP32 A与封存fixture逐项相同；训练seed变化不重置init_seed22 |
| reduction / SFT | 读取当前Orbit生产response切片与token reducer；microbatch/DP数学分片不变、零mask分母+1、首response/EOS/序列边界、T=1 |
| RL | G−1 sample std、交错group_index、全对/全错/singleton；零advantage组仍占token分母；缺失/错误ground truth预检失败；合法答案0可用 |
| optimizer / 恢复 | CPU Adam moments与计数、首步零LR、BF16有效参数与FP32 master恢复、连续/断点一致；这是CPU wrapper，不是Megatron TP2/DP2 fused optimizer认证 |
| freeze / endpoint | 实际production约束清掉目标optimizer梯度，参数/moments不变，另一因子继续且输入梯度不截断；真实D泄漏、完整线性forward、零/低秩basis拒绝填充 |
| teacher温度 | 实际production metadata adapter的T=.7、no_grad及冲突拒绝；SG多进程/CUDA服务仍需GPU验收 |
| GPU及正式结果 | **未运行**；不把CPU数学通过等同于新M6 100步、RL 300步或SFT有效性已经完成 |

C0的provenance与profile已落地；C1/C2的CPU数学/batch映射已检验；C3/C4的数据provider与损失入口已实现但真实数据manifest待填；C5的CPU恢复/freeze已检验，分布式恢复待GPU；C6提供可审阅的launch、endpoint评测接口和最小面板，不自动提交网格。

完整GPU gate需要同一真实batch的独立旧/新执行记录、预先固定的tolerance、误差分布、optimizer状态与实际D、运行时registry和短程resume。当前报告保留这些未验证项，不通过降低gate或改标签“完成”。SFT/RL manifest为unresolved时正式启动应当失败。
