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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json

import ray
from omegaconf import OmegaConf

from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.config import PPOConfig
from verl.trainer.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils import get_processor, get_tokenizer
from verl.workers.fsdp_workers import FSDPWorker
from verl.workers.reward import CustomRewardManager
import os
import debugpy

# local_rank = int(os.environ.get("LOCAL_RANK", 0))
# print(f"local_rank: {local_rank}")
# # 临时移除条件，总是启动debugpy
# # if local_rank == 0:  # 只让rank 0监听
# debugpy.listen(('0.0.0.0', 1116))
# print("Waiting for debugger attach on 1111...")
# debugpy.wait_for_client()


def main():
    cli_args = OmegaConf.from_cli() # 从命令行参数加载配置
    file_config = OmegaConf.load(cli_args.config)  # 从配置文件加载配置
    del cli_args.config

    default_config = OmegaConf.structured(PPOConfig()) # 创建默认配置
    ppo_config = OmegaConf.merge(default_config, file_config, cli_args) # 按照优先级从低到高合并配置
    ppo_config = OmegaConf.to_object(ppo_config) # 转换为Python对象

    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})


    ray.get(main_task.remote(ppo_config))
    # 调试模式：直接调用而不是远程执行
#     main_task_local(ppo_config)  # 直接调用本地版本
    


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config: PPOConfig):
    config.deep_post_init()
    print(json.dumps(config.to_dict(), indent=2))
    # instantiate tokenizer
    tokenizer = get_tokenizer(config.worker.actor.model.model_path)
    processor = get_processor(config.worker.actor.model.model_path, use_fast=True)

    # define worker classes
    ray_worker_group_cls = RayWorkerGroup
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(FSDPWorker),
        Role.Critic: ray.remote(FSDPWorker),
        Role.RefPolicy: ray.remote(FSDPWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    reward_fn = CustomRewardManager(
        tokenizer=tokenizer, num_examine=1, compute_score=config.worker.reward.compute_score
    )
    
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=None,
    )
    trainer.init_workers()
    trainer.fit()

def main_task_local(config: PPOConfig):
    """本地调试版本的main_task，运行在主进程中"""
    print("="*100)
    print("main_task_local (debug version)")
    config.deep_post_init()
    print(json.dumps(config.to_dict(), indent=2))
    # instantiate tokenizer
    tokenizer = get_tokenizer(config.worker.actor.model.model_path)
    processor = get_processor(config.worker.actor.model.model_path, use_fast=True)

    # define worker classes
    ray_worker_group_cls = RayWorkerGroup
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(FSDPWorker), # 生成动作和经验
        Role.Critic: ray.remote(FSDPWorker), # 评估状态价值
        Role.RefPolicy: ray.remote(FSDPWorker), # 参考策略（用于计算 KL 散度）
    }

    global_pool_id = "global_pool" # 资源池的名字
    resource_pool_spec = { # 定义资源池的规格，比如每个节点有多少GPU，一共多少节点。
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes, 
    }
    mapping = {  # 把三种角色都分配到同一个资源池。
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    reward_fn = CustomRewardManager(
        tokenizer=tokenizer, num_examine=1, compute_score=config.worker.reward.compute_score # 计算奖励 vision_reasoner
    )
    # 根据上面定义的资源池规格和角色分配，初始化资源池管理器。
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=None,
    )
    trainer.init_workers() # 初始化分布式工作器
    trainer.fit() # 开始训练


if __name__ == "__main__":
    main()
