set -x


PROJECT_NAME="dpwriter"
EXPERIMENT_NAME="expr_name"

MODEL_PATH="/SFT_CKPT_PATH/"
CKPTS_DIR=${CKPTS_DIR:-"SAVE_PATH/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
TRAIN_FILE=${TRAIN_FILE:-"DATA_PATH/train.parquet"}
TEST_FILE=${TEST_FILE:-"DATA_PATH/test.parquet"}

max_prompt_length=$((1024 * 1))
max_response_length=$((3072 * 1))
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 16))

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.project_name="$PROJECT_NAME" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=64 \
    trainer.save_freq=64 \
    trainer.total_epochs=5 \
    trainer.rollout_data_dir="${CKPTS_DIR}/rollout_data" \
    trainer.validation_data_dir="${CKPTS_DIR}/validation_data" \
    trainer.log_val_generations=64 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.use_div_branching=True \
    trainer.k_branches=32 \
    trainer.diversity_metric="ngram" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.shuffle=False \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    custom_reward_function.path=dpwriter_rewards/reward_skywork_w_div_norm.py \
    custom_reward_function.name=skywork_reward_w_div06 \
    reward_model.reward_manager=prime \
    reward_model.launch_reward_fn_async=True \
    algorithm.use_kl_in_reward=False \
    2>&1 | tee logs/"$PROJECT_NAME"_"$EXPERIMENT_NAME".log


# $@


