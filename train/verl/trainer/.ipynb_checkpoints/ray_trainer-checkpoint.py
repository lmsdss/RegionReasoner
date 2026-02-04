# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Dict, Optional, Type

import numpy as np
import torch
from codetiming import Timer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer import core_algos
from verl.trainer.config import PPOConfig
from verl.utils.rl_dataset import RLHFDataset, collate_fn
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import Tracking
from verl.workers.fsdp_workers import FSDPWorker

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

# 对每个样本的奖励分数应用 KL 散度惩罚（KL penalty），以防止新策略偏离参考策略太远，并动态调整 KL 系数。
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:] # 只保留响应部分的有效token

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys(): # 如果有参考策略，计算KL散度 逐 token
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value # 获取当前的KL系数
    else: # 如果没有参考策略，KL 惩罚为 0。
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)
    # 用 KL 惩罚修正奖励：奖励 = 原始分数 - β × KL 
    token_level_rewards = token_level_scores - beta * kld 
    # 计算当前 batch 的平均 KL 值（只统计有效 token）
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size) # 用当前 KL 更新 KL 控制器（自适应调整 β）
    data.batch["token_level_rewards"] = token_level_rewards # 把修正后的奖励写回 batch

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta} # 返回数据和本轮 KL 相关指标

    return data, metrics

# 根据不同的 advantage estimator（优势函数估计方法），计算每个样本的 advantage（优势）和 return（回报），用于 RL/PPO 算法的梯度更新
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group 支持多种 advantage 估计方法（如 GAE、GRPO、reinforce++、remax）
    # TODO: add other ways to estimate advantages
    if adv_estimator == "gae":
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards, values=values, eos_mask=response_mask, gamma=gamma, lam=lam
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "grpo": # 用 GRPO 算法，按分组（如同一 prompt）归一化奖励，计算优势和回报
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1) # 响应长度 2000
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "reinforce_plus_plus":
        token_level_rewards = data.batch["token_level_rewards"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "remax":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch["reward_baselines"]

        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=token_level_rewards, reward_baselines=reward_baselines, eos_mask=response_mask
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: Dict[str, Any]):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)

    return metrics

def _compute_response_info(batch: DataProto):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True):
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{
            f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
            for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy: # True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        if self.config.algorithm.adv_estimator == "gae":
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == "grpo": # True
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == "reinforce_plus_plus":
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == "remax":
            self.use_critic = False
        else:
            raise NotImplementedError

        self._create_dataloader()

    def _create_dataloader(self):
        
        # 判断数据路径是否为JSON格式  todo 0810
        is_json_format = self.config.data.train_files.endswith('.json')
        
        # 1. 创建数据集
        self.train_dataset = RLHFDataset(
            data_path=self.config.data.train_files, # data/VisionReasoner_multi_object_7k_840
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key, # 'problem'
            max_prompt_length=self.config.data.max_prompt_length, # 1300
            truncation="right",
            system_prompt=self.config.data.system_prompt, # 'You are a helpful assistant.'
            min_pixels=self.config.data.min_pixels, # 3136
            max_pixels=self.config.data.max_pixels, # 12845056
            is_json_format=is_json_format, # 判断数据路径是否为JSON格式  todo 0810

        )
        # use sampler for better ckpt resume
        # 2. 创建采样器
        if self.config.data.shuffle: # True
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        # 3. 创建数据加载器
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size, # 16
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        assert len(self.train_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else: # True
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and "wandb" not in self.config.trainer.logger:
            print("WARNING: `val_generations_to_log_to_wandb` is set, but no wandb logger is found.")
            return

        import wandb

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "pixel_values" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["pixel_values", "image_grid_thw", "raw_prompt_ids", "images"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = {"do_sample": False}

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            )

        self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool() # 1. 创建资源池
        #  2. 初始化资源池到类的映射字典
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine: # True
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout) # 3. 创建actor和rollout
            actor_rollout_cls = RayClassWithInitArgs(  # 创建ActorRollout类，包含初始化参数
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls  # 将类添加到对应资源池的映射中

        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:  # False
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy: # True
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model: # False
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        # 遍历每个资源池及其对应的工作进程类
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # 1. 创建共置（colocated）的工作进程类
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            # 2. 创建工作进程组
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            # 3. 启动工作进程组
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            # 4. 更新所有工作进程组
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            # 5. 保存工作进程组的引用
            self.wg_dicts.append(wg_dict)

        if self.use_critic: # False
            self.critic_wg: FSDPWorker = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy: # True
            self.ref_policy_wg: FSDPWorker = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg: FSDPWorker = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg: FSDPWorker = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: {save_checkpoint_path}/global_step_{global_steps}/actor
        local_global_step_folder = os.path.join(
            self.config.trainer.save_checkpoint_path, f"global_step_{self.global_steps}"
        )
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            self.global_steps,
            remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            self.critic_wg.save_checkpoint(
                critic_local_path,
                self.global_steps,
                remove_previous_ckpt=self.config.trainer.remove_previous_ckpt,
            )

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.save_checkpoint_path, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.load_checkpoint_path is None:
            return

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}")
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        logger = Tracking(
            project_name=self.config.trainer.project_name, # 'vision_reasoner'
            experiment_name=self.config.trainer.experiment_name, # 'run_visionreasoner_7b'
            default_backend=self.config.trainer.logger, # 
            config=self.config.to_dict(),
        )
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training 如果配置要求训练前做验证，则先评估一次模型。
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train: # False
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.val_only:
                return

        for _ in range(self.config.trainer.total_episodes): # 训练总轮数
            for batch_dict in self.train_dataloader:
                """
                    # batch_dict 包含：
                    # - input_ids: (batch_size, seq_len) - 输入token IDs
                    # - attention_mask: (batch_size, seq_len) - 注意力掩码
                    # - position_ids: (batch_size, 3, seq_len) - 位置编码
                    # - pixel_values: (batch_size, 3, H, W) - 图像像素值
                    # - image_grid_thw: (batch_size, 3) - 图像网格信息
                    # - problem: [str] - 问题文本
                    # - solution: [str] - 答案JSON字符串
                """
                self.global_steps += 1
                if self.global_steps >= self.training_steps: # 如果训练步数达到上限，则停止训练。 443 轮
                    break
                # 用于记录本轮训练的各种指标和耗时
                metrics = {} # 
                timing_raw = {}
                #  将原始batch数据转为DataProto对象，便于后续处理 batch_dict['pixel_values'][0].shape  batch_dict['problem']
                batch: DataProto = DataProto.from_single_dict(batch_dict) #
                # 分离张量和非张量数据
                # batch.batch.keys() # dict_keys(['input_ids', 'attention_mask', 'position_ids']))
                # batch.non_tensor_batch.keys() #  dict_keys(['pixel_values', 'image_grid_thw', 'id', 'problem', 'solution', 'image', 'img_height', 'img_width', 'images', 'raw_prompt_ids'])

                # non_tensor_batch: {'id': ['lvis_15266'], 'problem': ["'skateboard'"], 'solution': ['[{"bbox_2d": [105, 601, 356, 634], "point_2d": [336, 616]}]'], 'image': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=840x840 at 0x7F3A12E15DB0>], 'img_height': [640], 'img_width': [512]}
                # batch.batch["input_ids"].shape: (1, 1300)   batch.non_tensor_batch["solution"]: ['[{"bbox_2d": [105, 601, 356, 634], "point_2d": [336, 616]}]']
                # pop those keys for generation  针对有无图片输入，准备生成任务的输入batc
                if "pixel_values" in batch.non_tensor_batch.keys(): # True
                    gen_batch = batch.pop( # # 注意：problem 没有被 pop 到 gen_batch 中，它仍然保留在原始的 batch 中
                        batch_keys=["input_ids", "attention_mask", "position_ids"], # [1, 1300] [1, 1300] [1, 3，1300] 多维位置编码 使用MRoPE (Multi-dimensional RoPE) 3个维度分别代表：[t, h, w] - 时间、高度、宽度 
                        non_tensor_batch_keys=["pixel_values", "image_grid_thw", "raw_prompt_ids", "images"], # raw_prompt_ids 直接对原始提示进行token化，不添加特殊token，不padding 在vLLM推理时使用 input_ids：经过完整的处理流程，包括：添加特殊token，padding（左对齐） 生成attention_mask
                    ) # gen_batch.batch["input_ids"] gen_batch.non_tensor_batch["pixel_values"] 
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):  # wg: worker group
                        # 模型生成
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        """                            │
                            ├── [Ray分布式调用]
                            │
                            └── FSDPWorker.generate_sequences(prompts: DataProto)  ← 第一个
                                │
                                ├── 处理参数offload
                                ├── 添加meta_info (eos_token_id, pad_token_id)
                                ├── 进入rollout_sharding_manager上下文
                                │
                                └── self.rollout.generate_sequences(prompts=prompts)  ← 第二个
                                    │
                                    └── vLLMRollout.generate_sequences()
                                        ├── 使用raw_prompt_ids构建vLLM输入
                                        ├── 调用vLLM引擎生成序列
                                        └── 构建返回的DataProto
                        """
                    # 如果用remax算法，生成baseline输出并计算baseline奖励
                    if self.config.algorithm.adv_estimator == "remax": # False
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # 为每个样本分配一个唯一ID，用于后续的跟踪和调试。 按需重复batch，适应rollout
                    batch.non_tensor_batch["uid"] = np.array( # size 16
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object

                    ) 
                    # repeat to align with repeated responses in rollout 重复batch以适应rollout
                    batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    # 合并生成结果
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # self._balance_batch(batch, metrics=metrics) # TODO: re-enable balance batch

                    # compute global_valid tokens  统计每个样本的有效token数 size 32
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs 计算旧策略下的log概率，用于PPO算法
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                    # 如果使用参考策略，计算参考策略下的log概率
                    if self.use_reference_policy: # True
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values 如果有Critic，计算状态价值
                    if self.use_critic: # False
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_reward_model: # False
                            raise NotImplementedError

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch) # 计算奖励
                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.worker.actor.use_kl_loss:  # not grpo
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else: # True
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.worker.rollout.n,
                        )

                    # update critic
                    if self.use_critic: # False
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch) # 更新Critic网络参数

                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup 训练初期 只更新Critic  不更新Actor  让Critic学习准确的价值估计
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch) # Critic热身后，才开始更新Actor网络参数

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # validate 按频率做验证和保存模型
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

        # perform validation after training 训练结束后做一次最终验证并保存模型
        if self.val_reward_fn is not None: 
            val_metrics = self._validate()
            pprint(f"Final validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

        self._save_checkpoint()
