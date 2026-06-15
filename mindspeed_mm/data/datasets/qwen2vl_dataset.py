from dataclasses import field
import os
from typing import Optional
import warnings

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers.training_args import TrainingArguments

from megatron.training import get_args
from mindspeed_mm.data.data_utils.func_utils.convert import (
    DataArguments,
    DatasetAttr,
    load_tokenizer,
    align_dataset,
    SupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PackedSupervisedDatasetProcessor
)
from mindspeed_mm.data.data_utils.func_utils.log import get_logger
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.data.data_utils.func_utils.template import get_template_and_fix_tokenizer
from mindspeed_mm.data.action_tokenizer import ActionTokenizer, FASTTokenizer
from .data_qwen_packed import make_supervised_data_module_packed

logger = get_logger(__name__)


class DistributedIterableDataset(IterableDataset):
    def __init__(self, dataset, rank=None):
        args = get_args()
        self.data_parallel_size = args.data_parallel_size
        self.dataset = dataset
        self.rank = torch.distributed.get_rank() if rank is None else rank

    def __iter__(self):
        for idx, item in enumerate(self.dataset):
            if idx % self.data_parallel_size == self.rank % self.data_parallel_size:
                yield item


def get_qwen2vl_dataset(basic_param, preprocess_param, dataset_param):
    #if "cutoff_len" in basic_param.keys():
        #raise ValueError("`cutoff_len` is deprecated, please use `seq_length` instead.")
    data_args = DataArguments(**basic_param)
    if data_args.dataset_use:#加载离线pack的数据
        return get_qwen2vl_dataset_pack_offline(basic_param, preprocess_param, dataset_param)
    
    if not data_args.pack: #pack后的length要小于seq_legnth,否则seq_length太长时，一个样本中的图片太多，会导致显存溢出
        data_args.cutoff_len = get_args().seq_length
        
    process_args = ProcessorArguments(**preprocess_param)
    dataset_attr = DatasetAttr(**dataset_param["attr"])

    tokenizer_module = load_tokenizer(process_args)
    tokenizer, processor = tokenizer_module['tokenizer'], tokenizer_module['processor']
    setattr(processor, 'image_aug', process_args.image_aug)
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    # 确保主进程进行数据处理，其他进程复用缓存避免重复计算，该策略和llamafactory对数据处理策略一致
    with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # -----------------load dataset from file-------------------------------------------------------------------------
        train_dataset = load_dataset(path="json", data_files=data_args.dataset, split="train",
                                     cache_dir=data_args.cache_dir,
                                     streaming=data_args.streaming)
        if data_args.max_samples and not data_args.streaming:
            train_dataset = train_dataset.select(range(data_args.max_samples))

        val_dataset = None
        if data_args.val_dataset:
            val_dataset = load_dataset(
                path="json",
                data_files=data_args.val_dataset,
                split="train",
                cache_dir=data_args.cache_dir,
                streaming=data_args.streaming
            )
            if data_args.val_max_samples:
                val_dataset = val_dataset.select(range(data_args.val_max_samples))
            if data_args.val_rate is not None and data_args.val_rate > 0.0:
                warnings.warn(
                    "Warning: Both val_dataset and val_rate have been provided. The val_dataset will take priority, and the val_rate will be ignored.",
                    UserWarning)

        local_process_index = int(os.getenv("LOCAL_RANK", -1))
        if data_args.streaming:
            kwargs = {}
        else:
            kwargs = {
                "num_proc": data_args.preprocessing_num_workers,
                # 配置了overwrite_cache为false（默认为false)时，非rank0节点读取cache不再进行map处理
                # 配置了overwrite_cache为true（默认为false)时，所有节点都读取cache不再进行map处理
                "load_from_cache_file": (not data_args.overwrite_cache) or (local_process_index != 0)
            }
        logger.debug(f'Rank: %s, kwargs: %s', local_process_index, kwargs)
        # -----------------convert to sharegpt ---------------------------------------------------------------------------
        train_dataset = align_dataset(train_dataset, dataset_attr, data_args)
        if val_dataset:
            val_dataset = align_dataset(val_dataset, dataset_attr, data_args)

        # -----------------convert text to token id ----------------------------------------------------------------------
        if dataset_attr.ranking:
            dataset_processor_cls = PairwiseDatasetProcessor
        else:
            if data_args.pack:
                dataset_processor_cls = PackedSupervisedDatasetProcessor
            else: 
                dataset_processor_cls = SupervisedDatasetProcessor
        model_args = get_args().mm.model
        if model_args.action_tokenizer == 'ActionTokenizer':
            action_tokenizer = ActionTokenizer(tokenizer, model_args.action_start_token_id,
                                            bin_file=model_args.action_bin_file)
        elif model_args.action_tokenizer == 'FASTTokenizer':
            action_tokenizer = FASTTokenizer(tokenizer, model_args.action_start_token_id,
                                            fast_tokenizer_path=model_args.fast_tokenizer_path)
        else:
            raise NotImplementedError
        preprocess_func = dataset_processor_cls(template=template, tokenizer=tokenizer, 
                                                action_tokenizer=action_tokenizer, processor=processor,
                                                data_args=data_args).preprocess_dataset
        if data_args.streaming:
            train_dataset = train_dataset.map(
                preprocess_func,
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=(list(next(iter(train_dataset)).keys())),
                **kwargs,
            )
            train_dataset = DistributedIterableDataset(train_dataset)
            if val_dataset:
                val_dataset = val_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(val_dataset)).keys())),
                    **kwargs,
                )
                val_dataset = DistributedIterableDataset(val_dataset)
                return train_dataset, val_dataset
        else:
            train_dataset = train_dataset.map(
                preprocess_func,
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=(list(next(iter(train_dataset)).keys())),
                desc=f"Rank {local_process_index}, running tokenizer on train_dataset",
                **kwargs,
            )
            if val_dataset:
                val_dataset = val_dataset.map(
                    preprocess_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=(list(next(iter(val_dataset)).keys())),
                    desc=f"Rank {local_process_index}, running tokenizer on val_dataset",
                    **kwargs,
                )
                return train_dataset, val_dataset
        return train_dataset

def get_qwen2vl_dataset_pack_offline(basic_param, preprocess_param, dataset_param):
    if "cutoff_len" in basic_param.keys():
        raise ValueError("`cutoff_len` is deprecated, please use `seq_length` instead.")
    data_args = DataArguments(**basic_param)
    if not data_args.pack: #pack后的length要小于seq_legnth,否则seq_length太长时，一个样本中的图片太多，会导致显存溢出
        data_args.cutoff_len = get_args().seq_length
    process_args = ProcessorArguments(**preprocess_param)
    dataset_attr = DatasetAttr(**dataset_param["attr"])

    tokenizer_module = load_tokenizer(process_args)
    tokenizer, processor = tokenizer_module['tokenizer'], tokenizer_module['processor']
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    # 确保主进程进行数据处理，其他进程复用缓存避免重复计算，该策略和llamafactory对数据处理策略一致
    if True: #with TrainingArguments(output_dir='./').main_process_first(desc="pre-process dataset"):
        # -----------------load dataset from file-------------------------------------------------------------------------
        offline_dataset_args={
        "model_type": "qwen2.5vl",
        "dataset_use":data_args.dataset_use,#llava_instruct_150k, cambrian_737k
        "video_max_frames": 8,
        "video_min_frames": 4,
        "data_flatten": False,
        "data_packing": False,
        "base_interval": 2,
        "max_pixels":  28 * 28 * 576,
        "min_pixels": 28 * 28 * 16,
        "video_max_frame_pixels": 32 * 28 * 28,
        "video_min_frame_pixels": 4 * 28 * 28,
        }
        train_dataset=make_supervised_data_module_packed(tokenizer,offline_dataset_args)
    train_dataset = DistributedIterableDataset(train_dataset)
    return train_dataset
    
