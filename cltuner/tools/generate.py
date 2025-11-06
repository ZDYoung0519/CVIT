# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import json
import math
import os.path as osp
import tqdm
from types import FunctionType

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.logging import print_log
from mmengine.dist import (
    collect_results,
    get_dist_info,
    get_rank,
    init_dist,
    master_only,
)
from transformers import (
    GenerationConfig,
)

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint, prepare_inputs_labels_for_multimodal
from xtuner.registry import MAP_FUNC
from xtuner.utils.device import get_device, get_torch_device
from xtuner.tools.utils import get_stop_criteria, is_cn_string


def parse_args():
    parser = argparse.ArgumentParser(description="Eval model")
    parser.add_argument("config", help="config file name or path.")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file")
    parser.add_argument("--current-task", type=int, default=-1, help="current task index")
    parser.add_argument("--output-dir", type=str, help="checkpoint file")
    parser.add_argument("--num-chunks", default=1, type=int, help="checkpoint file")
    parser.add_argument("--chunk-idx", default=0, type=int, help="checkpoint file")
    parser.add_argument('--max-new-tokens', default=200, type=int)
    parser.add_argument(
        "--work-dir",
        help="the directory to save the file containing evaluation metrics",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    return args


def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)


def generate_answers(runner, dataset, args):
    device = get_device()

    # rank, world_size = get_dist_info()
    model = runner.model.module if hasattr(runner.model, "module") else runner.model

    # generate config
    tokenizer = dataset.tokenizer
    stop_words = ''
    stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=stop_words)

    args.max_new_tokens = 500 if not hasattr(args, "max_new_tokens") else args.max_new_tokens

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id,
    )

    rank = args.chunk_idx
    world_size = args.num_chunks
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)
    per_rank_ids = range(
        per_rank_samples * rank, min(n_samples, per_rank_samples * (rank + 1))
    )

    print(f"[CHUNK:{rank}][IDX:{str(per_rank_ids)}][LEN:{len(per_rank_ids)}][TOTAL:{n_samples}]")
    results = []
    for i in tqdm.tqdm(per_rank_ids, desc=f"Rank {rank}"):
        data = dataset[i]

        input_ids = data["input_ids"].to(device).unsqueeze(0)
        image = data["pixel_values"].to(device)

        visual_outputs = model.visual_encoder(
            image.unsqueeze(0).to(model.visual_encoder.dtype),
            output_hidden_states=True,
        )
        pixel_values = model.projector(
            visual_outputs.hidden_states[model.visual_select_layer][:, 1:]
        )
        mm_inputs = prepare_inputs_labels_for_multimodal(
            llm=model.llm,
            input_ids=input_ids.unsqueeze(0),
            pixel_values=pixel_values,
        )

        generation_output = model.generate(
            **mm_inputs,
            generation_config=gen_config,
            bos_token_id=tokenizer.bos_token_id,
            stopping_criteria=stop_criteria,
        )

        prediction = tokenizer.decode(generation_output[0], skip_special_tokens=True).strip()
        cur_result = {}
        cur_result["question_id"] = data["question_id"]
        cur_result["question"] = data.get("question")
        cur_result["answer"] = data.get("answer")
        cur_result["prediction"] = prediction

        results.append(cur_result)

            
    results = collect_results(results, n_samples)
    return results


def main():
    args = parse_args()

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # register FunctionType object in cfg to `MAP_FUNC` Registry and
    # change these FunctionType object to str
    register_function(cfg._cfg_dict)

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # disable visualizer
    cfg.visualizer = None
    
    # only build datasets that are to evaluate on
    cfg.test_dataset = cfg.test_dataset[args.current_task]
    cfg.test_dataloader['dataset'] = cfg.test_dataset

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # load checkpoint
    if args.checkpoint is not None:
        state_dict = guess_load_checkpoint(args.checkpoint)
        runner.model.load_state_dict(state_dict, strict=False)
        runner.logger.info(f"Load checkpoint from {args.checkpoint} successfully!")

    dataset = runner.test_dataloader.dataset
    data_name = dataset.metainfo['name']
    print_log("##########################################")
    print_log(f"############ {data_name} ############")
    print_log("##########################################")

    if not os.path.exists(args.output_dir) and get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # generate answers and save results
    output_file = os.path.join(args.output_dir, f"chunk{args.chunk_idx}.jsonl")
    results = generate_answers(runner, dataset, args)
    if get_rank() == 0:
        file = open(output_file, "w", encoding='utf-8')
        for res in results:
            file.write(json.dumps(res, ensure_ascii=False) + "\n")
        file.close()

if __name__ == "__main__":
    main()

