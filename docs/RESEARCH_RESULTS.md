# SuperMimic 算法研究结果

更新日期：2026-10-29。本文汇总当前工作区里已完成的复现诊断、理论分析和 toy/G1 实验。固定 checkpoint 上的短 rollout、模型误差和 toy 机制验证均不等同于多 seed 的完整训练收益。

## 结论

目前**没有证据证明某个新算法在完整 SuperMimic 训练中稳定优于原方法**。最可靠的产出是修复了 replay 终止标签错位、让 actor/critic 学习率真正独立，并确认 critic 的原始 MSE 随目标值尺度变化而变大。研究中有两个值得继续的方向：利用执行器的物理等价关系规范化世界模型输入，以及用独立随机截断估计真实跟踪回报。两者都尚未通过完整策略训练对照。

仓库现有复现基线在 README 中报告：BONES-SEED 全集归一化回报 0.930；LAFAN1 论文观测配置为 0.56–0.59，添加锚点相对观测的一次配置为 0.904。它们是复现/观测配置结果，不是本文算法改进带来的提升。

## 已修复的训练问题

**终止标签错位。** Replay buffer 每行的 `done` 表示该行转移到后继状态是否终止；窗口采样过去读取了下一行，令真实序列 `[0,0,1]` 变为 `[0,1,1]`。现在读取当前转移行。一次真实 G1 终止头对照中，留出集 7164 条转移含 22 个真实终止：正确标签的 Brier 误差为 .00114、阈值 .5 召回 14/22、非终止误报 3/7142；错位标签分别为 .00220、21/22、21/7142。错位标签增加了召回，也增加了误报；事件数少且实验只训练终止头，不能推断策略回报变化。

**独立学习率。** 训练循环曾用 actor 学习率覆盖 critic 的独立配置。现在 world model、actor 和 critic 使用各自基准学习率乘相同余弦调度因子，且旧的一参数 `set_lr` 调用保持兼容。被诊断的旧 LAFAN1 运行三者配置恰好都为 0.0002，因此这个配置错误不能解释那次 loss 曲线。

**critic 尺度诊断。** 从旧运行 TensorBoard 重算 50–2100 步共 42 个采样点，critic loss 与 critic target mean 平方的 Spearman 相关为 .766。1000 到 2000 步，loss 从 .00655 增至 .03437，目标均值从 .496 增至 2.230；`sqrt(loss)/abs(target_mean)` 这个粗略尺度比值则从 .163 降到 .083。loss 取最后一个训练 minibatch，target mean 是整批 rollout 的均值，所以该比值不是验证 RMSE；它只能说明 raw loss 上升本身不足以判断发散。训练现在额外记录 imagined target 标准差和更新前归一化 RMSE，指标名为 `policy/critic_target_std`、`policy/critic_preupdate_nrmse`；这是同批在线自洽诊断，不是独立验证。

终止标签消融的逐 seed 数字保留在[结果文件](research/results/g1_terminal_label_ablation.json)。生存概率下价值平移/缩放的目标恒等式由[toy 结果](research/results/mbrl_value_normalization_toy.json)数值验证；该恒等式不等于训练收益。

## 算法研究发现

**世界模型的物理输入等价性。** 当前无动作裁剪的关节位置控制里，实际目标为 $u=Sa+o-b$。因此同时改变动作 $a$ 和编码器偏置 $b$、保持 $Sa-b$ 不变时，物理目标不变。真实 G1 配对实验支持该一步等价关系；原模型对等价输入仍会给出不同预测。把模型输入改写为 $(a-S^{-1}b,0)$ 后，三个新随机种子的真实动作开环评估中，8 步 NSE 均值从 .18701 降至 .18447，32 步从 1.20458 降至 1.16552。

更强的重拟合对照显示，训练时规范化没有稳定优于“原始训练、规范化推理”：32 步 NSE 为 .84248 对 .84396，8 步 .223455 对 .223443；G1 动作响应对照也没有稳定优势。有限策略方向扰动在 9 个新起点中 4 个回报更高，平均差 +.01438，但共享模型权重，起点和种子少，并非多轮策略训练。物理等价关系的开环误差、重拟合和动作响应结果见[汇总数据](research/results/g1_bias_canonical_holdout_summary.json)、[重拟合审计](research/results/g1_bias_retrain_confirm_audited.json)、[响应汇总](research/results/g1_bias_retrain_response_summary.json)和[方向审计](research/results/g1_bias_actor_confirm_audited.json)。

**随机几何截断的回报估计。** 固定 critic $V$，独立地以 $P(T>t)=q^t$ 随机截断真实轨迹，定义

$$\widehat G=V(s_0)+\sum_{t=0}^{T-1}(\gamma/q)^t\{r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)\}.$$

在 $T$ 与轨迹随机性独立、期望可交换且适当可积的条件下，TD 项望远镜相消，$E[\widehat G]$ 等于该策略的真实折扣回报；critic 不必准确，但估计方差可能很大。独立 toy 随机流符合推导，数值结果见[随机截断 toy 复测 9820](research/results/mbrl_roulette_value_confirm_9820.json)和[9830](research/results/mbrl_roulette_value_confirm_9830.json)。G1 留出集 36 个起点的交叉误差均值中，训练数据拟合出的 affine critic 候选相对原 critic 的 pooled 结果为 .9809 对 1.5575；三个留出 seed 分别为 .7759 对 .4960、1.6366 对 2.3698、.5304 对 1.8068。第一个 seed 变差，样本仅来自三个动作片段，交叉误差可以为负，不能据此宣称校准改善或训练收益。完整结果见[汇总数据](research/results/g1_roulette_value_holdout_summary_9860_9862.json)和[逐轨迹审计](research/results/g1_roulette_value_long_audit_9850_9862.json)。

**误差指标不能替代策略收益。** 同参考帧配对估计曾把 G1 梯度批间方差降到多起点 critic-score 基线的约 .04–.06 倍，但后续固定范数真实回报比较没有稳定胜出。删除策略目标的末端价值项也没有稳定收益；理论上，即使末端价值准确，去掉它也可能让短期梯度与完整目标方向相反。因此不能仅凭更低的模型误差、梯度方差或 critic loss 选定改进。

## 当前判断与下一步

终止标签和学习率修复已经进入训练代码；新尺度指标可在之后的训练日志中观察。物理输入规范化与随机截断估值保留为研究候选，尚未合并为新的默认算法。下一步应减少并行探索，优先用相同数据、计算预算和多个独立训练 seed，对一个候选做完整训练，再用未见动作的确定性 clip 评估回报、失败率和模型误差；候选若不能稳定改善真实表现，就放弃。

复现实验的源代码位于 `scripts/analysis/`，必要回归/机制测试位于 `tests/`。G1 实验还需要本地 LAFAN1 数据和训练 checkpoint；它们不包含在 GitHub 提交中。

为减小仓库并清理中间产物，保留最终汇总 JSON、研究代码和少量机制/回归测试，删除逐步 rollout 张量、梯度向量及大量单次探针输出。文中的统计汇总仍可查阅；逐样本审计需要用原始数据和 checkpoint 重新采集。
