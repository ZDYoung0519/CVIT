# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionModel,
)

from mmengine.dataset import DefaultSampler
from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE

from xtuner.dataset import ConcatDataset
from cltuner.dataset.evalutaion import BaseEvalDataset
from cltuner.engine.runner.loops import TestLoop
from cltuner.engine.hooks import SetTaskHook

#######################################################################
#                          PART 1  Settings                           #
#######################################################################

# LLaVA v1.5 model with pre-trained and instruction-fintuned weights
llm_name_or_path = "/storage/huggingface/llava-v1.5-7b-xtuner"      # llm weights
pretrained_pth = "/storage/huggingface/llava-v1.5-7b-xtuner/mm_projector_xtuner.pt"     # mlp weights
visual_encoder_name_or_path = "/storage/huggingface/openai/clip-vit-large-patch14-336"

# Data
data_root = "/storage/public_datasets/UCIT/UCIT/"
image_folder = "/storage/public_datasets/"
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = 2048

# Scheduler & Optimizer
batch_size = 8  # per_device
accumulative_counts = 1
dataloader_num_workers = 4
max_epochs = 1

optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03

# Save
save_steps = 2000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
)

model = dict(
    type=LLaVAModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    pretrained_pth=pretrained_pth,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
    ),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
    )
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
SYSTEM = ""

train_dataset = [
    dict(
        type=LLaVADataset,
        data_path=data_root + "ImageNet-R/train.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset_map_fn=llava_map_fn,
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=LLaVADataset,
        data_path=data_root + "ArxivQA/train_4w.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset_map_fn=llava_map_fn,
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=LLaVADataset,
        data_path=data_root + "VizWiz/train.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset_map_fn=llava_map_fn,
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=LLaVADataset,
        data_path=data_root + "IconQA/train.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset_map_fn=llava_map_fn,
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=LLaVADataset,
        data_path=data_root + "CLEVR-Math/train_4w.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset_map_fn=llava_map_fn,
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=LLaVADataset,
        data_path=data_root + "Flickr30k/train_brief_4w.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset_map_fn=llava_map_fn,
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        max_length=max_length,
        pad_image_to_square=True,
    )
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=dict(type=ConcatDataset, datasets=train_dataset),
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=default_collate_fn),
)

test_dataset = [
    dict(
        type=BaseEvalDataset,
        metainfo=dict(
            name='ImageNet-R', 
            anno_file=data_root+"ImageNet-R/test_3000.json", 
            eval_script='cltuner/tools/eval/eval_deepseek_r1.py'
        ),
        data_path=data_root+"ImageNet-R/test_3000.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        system=SYSTEM,
        prompt_template=prompt_template,
        max_length=max_length,
        pad_image_to_square=True,
    ),
    dict(
        type=BaseEvalDataset,
        metainfo=dict(
            name='ArxivQA', 
            anno_file=data_root+"ArxivQA/test_3000.json", 
            eval_script='cltuner/tools/eval/eval_deepseek_r1.py'
        ),
        data_path=data_root + "ArxivQA/test_3000.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        system=SYSTEM,
        prompt_template=prompt_template,
        max_length=max_length,
        pad_image_to_square=True,
    ),
        dict(
        type=BaseEvalDataset,
        metainfo=dict(
            name='VizWiz',
            anno_file=data_root+"VizWiz/val_coco_type_3000.json",
            eval_script='cltuner/tools/eval/eval_caption.py'
        ),
        data_path=data_root + "VizWiz/test_3000.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        system=SYSTEM,
        prompt_template=prompt_template,
        max_length=max_length,
        pad_image_to_square=True,
    ),
        dict(
        type=BaseEvalDataset,
        metainfo=dict(
            name='IconQA',
            anno_file=data_root+"IconQA/test_3000.json", 
            eval_script='cltuner/tools/eval/eval_deepseek_r1.py'
        ),
        data_path=data_root + "IconQA/test_3000.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        system=SYSTEM,
        prompt_template=prompt_template,
        max_length=max_length,
        pad_image_to_square=True,
    ),
        dict(
        type=BaseEvalDataset,
        metainfo=dict(
            name='CLEVR-Math',
            anno_file=data_root+"CLEVR-Math/test_3000.json", 
            eval_script='cltuner/tools/eval/eval_deepseek_r1.py'
        ),
        data_path=data_root + "CLEVR-Math/test_3000.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        system=SYSTEM,
        prompt_template=prompt_template,
        max_length=max_length,
        pad_image_to_square=True,
    ),
        dict(
        type=BaseEvalDataset,
        metainfo=dict(
            name='Flickr30k',
            anno_file=data_root+"Flickr30k/val_coco_type_3000.json", 
            eval_script='cltuner/tools/eval/eval_caption.py'
        ),
        data_path=data_root + "Flickr30k/test_3000.json",
        image_folder=image_folder,
        tokenizer=tokenizer,
        image_processor=image_processor,
        system=SYSTEM,
        prompt_template=prompt_template,
        max_length=max_length,
        pad_image_to_square=True,
    )
]

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(type=ConcatDataset, datasets=test_dataset),
    collate_fn=dict(type=default_collate_fn),
)


#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="float16",
    paramwise_cfg=dict(custom_keys={'projector.model': dict(lr_mult=0.1)})
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
test_cfg = dict(type=TestLoop)
test_evaluator = dict()

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer),
    dict(type=SetTaskHook)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)


