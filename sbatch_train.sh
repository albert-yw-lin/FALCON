#!/bin/bash
#
#################
# SLURM RESOURCE ALLOCATION
#################
#SBATCH --partition=viscam,svl
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=viscam
#SBATCH --mem=100G
#SBATCH --job-name="g1_29dof_falcon"
#
# Time limit: 2 days (format is DAYS-HOURS:MINUTES:SECONDS)
#SBATCH --time=2-00:00:00
#
# Output and Error logs (The %j will be replaced by the unique Job ID)
# Make sure the 'logs' folder exists in your directory: mkdir -p logs
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Notifications
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ywlin@stanford.edu

#################
# ENVIRONMENT SETUP
#################
# Print info for debugging
echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Current directory: $(pwd)"

# Initialize Conda (standard path for Sherlock/Stanford clusters)
# If this fails, run 'which conda' and update the path below
source /viscam/u/ywlin/miniforge3/etc/profile.d/conda.sh

# Activate your environment
conda activate fcgym

# Move to the project root
cd /svl/u/ywlin/FALCON

#################
# EXECUTION
#################
# We use the backslash \ to break the long command into multiple lines for readability
python humanoidverse/train_agent.py \
    +exp=decoupled_locomotion_stand_height_waist_wbc_diff_force_ma_ppo_ma_env \
    +simulator=isaacgym \
    +domain_rand=domain_rand_rl_gym \
    +rewards=dec_loco/reward_dec_loco_stand_height_ma_diff_force \
    +robot=g1/g1_29dof_waist_fakehand \
    +terrain=terrain_locomotion_plane \
    +obs=dec_loco/g1_29dof_obs_diff_force_history_wolinvel_ma \
    num_envs=4096 \
    project_name=g1_29dof_falcon \
    experiment_name=g1_29dof_falcon \
    +opt=wandb \
    obs.add_noise=True \
    env.config.fix_upper_body_prob=0.3 \
    robot.dof_effort_limit_scale=0.9 \
    rewards.reward_initial_penalty_scale=0.1 \
    rewards.reward_penalty_degree=0.0001

echo "Job finished at: $(date)"
exit 0