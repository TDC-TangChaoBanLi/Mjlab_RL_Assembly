from mjlab_rl_assembly.peg_env_cfg import peg_ppo_runner_cfg, peg_env_0_cfg


from mjlab.tasks.registry import register_mjlab_task


register_mjlab_task(
  task_id="Mjlab-Peg",
  env_cfg=peg_env_0_cfg(),
  play_env_cfg=peg_env_0_cfg(play=True),
  rl_cfg=peg_ppo_runner_cfg(),
)