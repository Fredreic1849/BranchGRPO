## More Discussion on FLUX
1. We set the rollout inference batch size to 1 because we observed differences in probability outputs when it exceeds the training batch size. This inconsistency may stem from PyTorch or other underlying mechanisms, though the exact cause remains unclear.
2. A stronger supervised fine-tuning (SFT) stage can suppress exploration during the GRPO phase. For those developing custom models, we recommend using fewer SFT iterations to maintain exploratory diversity.
3. For some extreme cases-such as when using the same prompt and initial noise but your custom reward model fails to distinguish between different trajectories, you can experiment with different initial noise values within a prompt. Actually, if the reward model can effectively distinguish trajectories, using the same initial noise is preferable for stability.
4. The provided reward curves show FLUX's performance with extended training (using a larger ```max_train_steps```). However, extended training does not improve visualization quality, likely due to limitations in the reward model design. Since HPS-v2.1 was not optimized for FLUX, its effectiveness is constrained. But we don't have better open-source reward models. Enabling EMA (Exponential Moving Average) will enhance visualization.

<div align="center">
<img src=../assets/rewards/opensource_flux_more_steps.png width="49%">
<div>

