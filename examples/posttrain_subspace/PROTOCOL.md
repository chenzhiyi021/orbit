# Orbit 后训练子空间实验：实施协议 v1

日期：2026-10-16。依据：用户提供的《Implementation audit：M6 LoRA-r2 与 zh RL LoRA-r4》及本轮已作出的研究选择。

**状态：研究设计可冻结；新分支的迁移和实际运行仍须验收。** 本文件不表示已经创建或推送分支，没有新训练，也没有重新执行audit引用的枚举测试。

## 1. 本轮定案

| 协议 | 数据／用途 | 目标训练预算 | 学习率 | 地位 |
|---|---|---|---|---|
| M6 OPD | 当前封存的mixed B_target；Qwen3-1.7B ← Qwen3-4B-Instruct-2507 | P256×G1；每rollout一次update；100次optimizer事务 | r2峰值3.2e-4；保留真实10步warmup＋cosine的逐步序列 | 迁移基准与首轮干预主线 |
| Official-CoT SFT | Zhiyi已取得有效增益的OpenMathReasoning CoT处理流程 | 本轮计划B256、100次update；完整response监督口径 | warmup＋cosine；具体LoRA LR须完成一次有效性校准后锁定 | SFT主实验，不以较弱的same-teacher训练替代 |
| Teacher-matched SFT | 与M6共同teacher/prompt的冻结回答 | 按需建立B256、100步控制 | 独立标识的SFT配置 | 数据来源／协议控制，不先扩大实验 |
| Math RLVR | zh分支的数学verifier与GRPO逻辑 | P64×G4；每rollout一次update；300次optimizer事务 | **显式LR=2e-4、constant、warmup=0**，用于r4候选 | RL主实验；2e-4来自用户已选配方，不是audit认证的历史执行值 |

统一源码、Batch定义、有效参数空间接口、归约、checkpoint、日志、评测。**不把不同数据、温度、初始化、schedule和预算伪装成纯objective对照。** 若日后需要排序SFT／OPD／RL的敏感度，再补共同任务、预算和初始化的有针对性控制。

本版替代此前unified100计划中以下默认：RL全都100步；RL使用population std；将M6 response默认写成4096；把M6暂称为centered OPD；SFT只以same-teacher rollout为主。

## 2. 证据层级

- **M6运行证据**：100步、25,600 prompts、107,622,649 response tokens、8rank最终Adam step=100；544个封存文件当前哈希一致。基础Orbit commit为c5a58abae6f97937083d522bdf668dc74cc9d02e，实际还依赖封存hook及worktree改动。完整启动时diff及部分未封存core文件的历史字节仍未确认。
- **RL代码证据**：be4058abb805ee4abbc5cbf526c543a5352b414d／zh/subspace-opd-rl，rank覆盖到4后的静态路径。没有成功run的resolved dump、数据hash、实际LR或optimizer状态。因此不将它等同于Hao此前高分run的完全复现。
- **SFT**：本次audit没有覆盖其训练实现。采用官方CoT是已确定的研究选择；原成功配置与统一LoRA配置的关系仍要登记。
- audit报告枚举测试通过，固定prefix的RKL梯度最大绝对误差1.6691960e-7。本轮只核读报告，没有取得并重新执行其工作区脚本，不把该值当作本轮独立复算。

## 3. M6数学与运行协议：保留，不重新发明

### 3.1 核心形式

令p_theta为T=0.7的learner，q为T=0.7的固定teacher。A_it=0.1*stopgrad(log q - log p_old)，无额外baseline、whitening或group normalization。ratio为

    r = exp(logp_current - stopgrad(logp_current))

其前向数值恒为1，仍保留score梯度。保留原clipped helper以通过迁移回归，但此配置的[0.8,1.2] clip不触发。不要把ratio替换成常数1，也不要未经回归替换成前向数值不同的-A*logp实现。

原梯度为全局token平均的

    -0.1 * stopgrad(log q - log p_old) * grad(log p_current).

可准确描述为：**同步fresh rollout、单次update、未中心化的sampled-token reverse-KL policy-gradient surrogate，reward乘0.1，全局token归约。**

它在固定prefix、behavior确为p、完整support、无active clipping的条件下对应0.1倍conditional RKL梯度。它不是完整自回归sequence KL的已认证无偏梯度，不是OPD² delta-distillation，也不是官方整套recipe的逐项复现。

### 3.2 必须锁定的字段

| 字段 | 值 |
|---|---|
| rank / alpha / scale | 2 / 2 / 1 |
| A初始化 | name-seeded FP32 uniform [-1/sqrt(n),1/sqrt(n)]；不是Xavier |
| B初始化 | master和BF16都严格零 |
| 初始化与训练随机性 | init_seed=22的per-name规则；training/data seed42；sample seed42+canonical_row。三个概念不得合并 |
| 参数布局 | 112个fused模块、224个独立A/B张量；QKV共享A、gate/up共享A，o/down独立 |
| 独立可训练参数 | 1,835,008；全部为A/B；其余base/embedding/head/norm冻结 |
| 数据 | 封存B_target.jsonl，hash bfebe57eb0b3bccafcfa7efed981ed946f6af6192db363c661f19c46f78a1e0d；原prompt IDs/顺序/模板不改 |
| P / G / B | 256 / 1 / 256 |
| update复用 | 每rollout只有1次optimizer update；32个microbatch属于梯度累积 |
| prompt / response / context cap | **4096 / 8192 / 12288** |
| sampling | T=.7, top-p=1, top-k=-1；全词表support |
| teacher评分 | 同一sample.tokens，prefill T=.7；实际hook比HTTP payload字段优先 |
| learner old | update前同版本Megatron no_grad重算；不是SGLang behavior logp |
| temperature | behavior、teacher、old learner、current learner均.7 |
| advantage | sampled teacher-old_learner，乘.1，全detach |
| centering / whitening / TIS / OPSM / ICEPOP | 全关闭；无新增score centering |
| KL reward / KL loss / entropy | 全0 |
| reduction | 全局response token mean，不是sequence mean |
| mask | 保留已生成的response tokens，包括8192处截断的前缀；无假EOS |
| Adam | beta1=.9, beta2=.999, eps=1e-8, weight_decay=0 |
| gradient clip | 全局L2阈值1；历史该run最大preclip norm=.0342436284，未触发 |
| runtime | TP1/DP8/PP1/CP1；复制式FP32 master Adam；BF16计算；TF32关闭；保留已审计backend设置 |

M6 behavior/learner sampled logprob mean-abs≤.02、RMS≤.05是本run的原验收阈值，实际跨步最大为.01514434/.04766641。保留gate并记录分布；它不证明两引擎的完整分布完全一致。

强制ratio=1的profile只接受同步fresh rollout、单次update。要增加多epoch、重放或async，必须使用另一个显式protocol ID，重新说明ratio和校正；不能沿用本profile而不改声明。

### 3.3 真正的LR trace与计步

第s次optimizer事务执行前，u=s-1。实际LR：

- 0≤u<10：3.2e-4*u/10。
- 10≤u≤99：3.2e-5 + (3.2e-4-3.2e-5)/2 * [1+cos(pi*(u-10)/90)]。

| optimizer step | applied LR |
|---:|---:|
| 1 | 0 |
| 2 | 3.2e-5 |
| 10 | 2.88e-4 |
| 11 | 3.2e-4 |
| 100 | 3.208772090925022e-5 |

scheduler在step100后输出3.2e-5，不是第100步的applied值。**不改变这个一位时序，也不为了凑100次非零权重移动额外训练一步。** 第1步更新moments/Adam计数而权重不变；历史第2步B改变，第3步A/B均改变。

因此D1=0，不存在有学习含义的D1-SVD basis。D0/D1的方向相似度记为undefined；使用G0或第2步非零更新必须明确改名。保存optimizer事务编号、rollout编号、applied LR与实际参数位移，不统一用模糊step标签。

constant控制按相同真实warmup时序构造，第11次update开始平台不衰减。优先从共同step10完整状态分叉，先只对一个有效M6条件做；与cosine不同的累计LR和实际移动量均记录。

## 4. RL协议：接受代码定义，显式改LR，不称作高分旧run已认证

### 4.1 最终采用的算法字段

- math-only verifier；提取最后boxed，归一化／符号等价判定，正确1错误0。
- P=64、G=4、B=256；组统计在DP分发前按group_index计算。
- **样本标准差，分母G-1=3**；epsilon1e-6；不改成此前建议的population std。
- 序列advantage broadcast到该response所有token；无额外.1倍数、future-return、GAE或token whitening。
- token ratio=exp(current_learner-old_learner)，clip[.8,1.2]；force_on_policy_ratio=False。
- T=1、top-p=1、top-k=-1；teacher为空；无reference-KL和entropy，无TIS等校正。
- 每rollout只做一次global batch update；全局token平均。
- 全对／全错组保留、advantage为0，也保留在token分母；不动态重采样。零新梯度不保证Adam已有moment不会移动参数。
- response cap8192。**prompt/context cap没有被原脚本实际锁定，必须在运行前补齐且保留数据过滤记录**，不要声称旧run也是4096/12288。
- LoRA r4/alpha4，shared-A；A=Xavier normal、B=0；预期独立参数3,670,016，须以实际registry验收。
- **配置显式LR=2e-4，constant300步，warmup=0。** 原脚本可执行默认是5e-6，注释的5e-5也不采用。该选择来自用户已有经验，是本轮采用的配置，不是audit对成功run的历史认证。

命名：无KL、全局token归约的token-level GRPO/PPO。不要因为token averaging就重新命名为完整DAPO或Dr.GRPO。

### 4.2 Runtime整合

统一branch以M6封存运行环境为锚点，移植zh的verifier／group／loss逻辑，不把zh的Bridge/MCore/SGLang依赖整套覆盖M6。

第一版RL启动profile保留已读代码的TP2/DP2、动态microbatch与Xavier语义，在共同依赖栈中通过fixed-batch回归和实际运行。M6仍TP1/DP8。两种profile不是纯objective对照；后续若统一成TP1/DP8或Kaiming，单独命名、验收，不在空间干预时同时改变。

所有运行必须新增开启non-finite检测；数据预检不得将缺失expected_answer当普通答错。不能继承原CLI的--no-check-for-nan-in-loss-and-grad。对合法有限样本这不应改变数学结果；发现异常时中止并记录，不静默过滤继续凑预算。

移除研究profile中的--no-save-optim与--no-save-rng，保存未来冻结/继续训练所需状态。distributed optimizer与复制式optimizer的差异记入runtime profile，需验证数学归约、参数去重与恢复，不宣称跨kernel逐位一致。

300步预算为19,200次prompt抽取、76,800条response，唯一题数另算。不能与M6的25,600条response称为同总compute。

## 5. SFT的定案与剩余最小补录

Official-CoT SFT为主，teacher-matched SFT为后续控制。主协议计划在统一student/adapter框架下运行100步、B256、全局token mean；SFT loss temperature=1。生成teacher回答的temperature不得传给CE。

**本次audit没有SFT配置。以下字段不从M6/zh猜：**

1. 官方CoT的revision/subset、CoT/TIR选择、去重/正确性过滤、按问题隔离的Train/Dev/Eval。
2. 成功run的prompt格式、think tags、assistant prefill、response/context上限与截断规则。
3. 成功run的原参数化与LR；统一LoRA的rank/初始化/scale及LR需要校准，不直接沿用FullFT的LR。
4. optimizer与packing/reducer验证；所有真实response预测位置都参与，prompt/padding不参与，不添加假EOS。
5. 正式task evaluation manifest。

研究实现首次以r2为候选、r8按需确认；若原已成功SFT使用不同参数化，保留其回归配置，不把旧分数直接登记在新LoRA行。给统一LoRA一次小范围Dev校准，再冻结LR进入空间干预。主schedule选择warmup10＋cosine到.1峰值，是本轮方案选择，不是该audit认证的历史SFT事实。

长度先保留成功Official-CoT流程，不为凑M6 cap而默默裁短。若改变长度，明确作新profile并记录完整答案覆盖率。主SFT以task和同源held-out NLL评估，Qwen teacher-bank KL仅作共同诊断。

## 6. 共用实现与数值契约

### 6.1 共用Batch与reducer

每条样本显式包含prompt_id、group_id、token_ids、response_mask、自然终止/截断标志、policy_version、所用temperature。各类logprob字段独立，不能复用一个含义不定的log_probs。

旧reducer在每条全mask=0样本上给分母加1：N=sum_i max(n_valid_i,1)。本轮先保留reference回归；要求主训练有效样本n_valid>0并记录全mask=0数量。若正式要改变其语义，单列版本，不悄悄修复。RL零advantage组通常mask仍为1，与全mask=0不是一回事。

### 6.2 冻结与空间约束

- freeze-A0使用原始A0，不先正交化或缩放；仍允许梯度通过模块输入流向其他可训练层。
- 从共同checkpoint冻结时保持共同初始函数、未被冻结因子的moments、scheduler/RNG/数据游标。免费继续分支也通过同一restart过程。
- 固定空间与Haar比较使用相同系数维数、固定因子尺度与模块布局；不把native rank与固定因子rank相同当作同可训练预算。
- 精确约束使用前向参数化，核查实际D禁区能量；不只投影gradient/momentum后声称满足约束。
- 最终r2空间最多提供2个非零方向；要提取top16或top256区域，需有足够rank的dense来源，不补零空间向量。

### 6.3 提交与恢复

各协议保存model/tokenizer/data/template hash；init_seed、train_seed、basis_seed和eval_seed分别记录。当前M6复现的init_seed=22必须保留，不能因为train_seed=42就改成42。

保存adapter master与optimizer状态，不要求每个小adapter checkpoint重复保存整份冻结base。M6至少完整保存0/10/20/40/60/80/100，轻量几何保存点保留原15个时点；RL至少0/20/50/100/150/200/250/300。需要的新增分叉点也保存完整状态。正式评测与保存频率分开。

## 7. 正式实验顺序

### A. 迁移验收与有效基线

1. 固定真实batch比较迁移前后M6的mask、温度logprob、raw advantage*.1、scalar loss、grad、Adam状态、实际D与LR trace。
2. 迁移后完成一条M6 r2 100步；它是新增run，不要求随机生成分数逐位复制45.15。关键比较再补独立train seeds。
3. RL先做固定batch组统计／归约／梯度回归，填齐数据与长度，显式LR=2e-4，实际跑一条300步并完整记录registry与reward曲线。
4. SFT补齐manifest，在统一实现上完成有效100步基线。不是先训练一大批受限模型再发现自由基线没学起来。

### B. 第一轮M6干预

只先做：Free r2、完全相同A0从头冻结、共同step20的Free-resume与Freeze-A-resume，结束于总100步。后验reset-A与初始空间投影作为解释，不冒充中途冻结。若需要判断输出侧，再加同checkpoint的Freeze-B分支；A/B冻结剩余参数不同，不用于同预算左右优劣排序。

冻结结果有意义后，在恒定LR上复查一个关键对照。新的constant分支保持warmup trace与总100步，不叫等累计LR实验。

### C. 空间选择与跨协议确认

在一个能正常学习的固定因子预算上，比较参考／纯Haar／能量匹配随机。给双方同等Dev调参机会并记录来源总成本。已知成功终点空间先验证完整forward重构，再从零重训；表示能力与有限预算优化分开。

不要第一轮跑来源×rank×侧向×schedule×objective的笛卡尔积。RL先完成300步有效基线，再安排其自身时间点的冻结；SFT同理。不预设SFT>OPD>RL顺序。math-only桥接在需要纯协议结论时再做。

## 8. 评测与成本

- M6和SFT正式终点100，RL正式终点300；共享可比的数学eval manifest时仍不叫等预算性能比较。
- 完整数学评测主要在终点做；训练中小Dev用于有效性/超参，不用Eval挑checkpoint。SFT可用较便宜NLL，M6用固定bank KL，RL记录reward及有效组比例。
- 保存逐题生成及judge输出，训练seed与重复生成次数分开。记录response length、truncation、token数、wall time、GPU-hours、teacher评分成本。
- 同一冻结对照以相同总训练长度结束；源basis费用与目标训练费用分列。
- 训练reward稳定不是held-out质量收敛。固定100/300步是预算定义，不是数学意义的全局收敛认证。

## 9. 最后还需完成的验收，不再改变研究方向

1. RL数据ground truth与base/tokenizer身份、明确prompt/context cap，以及**运行时**LR=2e-4/registry/group semantics。
2. SFT成功Official-CoT数据处理与统一LoRA配置。
3. 新分支M6固定batch/optimizer迁移回归、实际短程GPU校验、保存与resume。
4. 所有split所需optimizer/RNG完整，non-finite和禁区leakage核查通过。

这几项是运行前门槛，不要求先找齐所有历史pip freeze或等待所有外部讨论结束。旧历史字节缺口如实保留；新分支从现在开始完整封存。
