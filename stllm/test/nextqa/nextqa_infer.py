import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import json
from tqdm import tqdm
from nextqa import NextQA_dataset, infer_NextQA, check_ans
import argparse
import os

from stllm.common.config import Config
from stllm.common.registry import registry
# imports modules for registration
from stllm.datasets.builders import *
from stllm.models import *
from stllm.processors import *
from stllm.runners import *
from stllm.tasks import *

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt-path", required=True, help="path to checkpoint file.")
    parser.add_argument("--anno-path", required=True, help="path to NextQA annotation.")
    parser.add_argument("--num-frames", type=int, required=False, default=100)
    parser.add_argument("--specified_item", type=str, required=False, default=None)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--system_llm", action='store_false')
    parser.add_argument("--ask_simple", action='store_true')
    parser.add_argument("--use_16frames_padding", action='store_true')
    parser.add_argument("--use_32frames_padding", action='store_true')
    return parser.parse_args()

def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model

    print('Initializing Chat')
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"args: {args}")

    cfg = Config(args)
    print(f"Config: {cfg}")
    
    model_config = cfg.model_cfg    
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = args.ckpt_path
    print(f"model_config: {model_config}")

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    for name, para in model.named_parameters():
        para.requires_grad = False
        print(f"Name: {name}, Shape: {para.shape}, Parameter: {para.numel()}")

    model.eval()
    
    all_token = ~(model_config.video_input=='mean')
    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    videos_len = []
    dataset = NextQA_dataset(args.anno_path, num_segments=args.num_frames, resolution=224, specified_item = args.specified_item)
    for example in tqdm(dataset):
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1

        # question_prompt='', # add in the end of question
        # answer_prompt=None, # add in the begining of answer
        # return_prompt='',  # add in the begining of return message
        pred = infer_NextQA(
            model,example, 
            system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n",
            question_prompt="\nAnswer with the option's letter from the given choices directly.",
            answer_prompt=None,
            return_prompt='',
            system_llm=args.system_llm,
            all_token=all_token,
            ask_simple=args.ask_simple,
            use_16frames_padding=args.use_16frames_padding,
            use_32frames_padding=args.use_32frames_padding,
        )

        gt = example['answer']
        if args.specified_item:
            res_list.append({
                'video_path': example['video_path'],
                'question': example['question'],
                'pred': pred,
                'gt': gt,
            })
        else:
            res_list.append({
                'pred': pred,
                'gt': gt
            })
        assert pred[0].isupper(), f"Error: {pred}"
        assert gt[0] == '(' and gt[2] == ')' and gt[1].isupper(), f"Error: {gt}"
        # if check_ans(pred=pred, gt=gt[1]):
        if pred[0].lower() == gt[1].lower():
            acc_dict[task_type][0] += 1
            correct += 1
        print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
        print(f"Total Acc: {correct / total * 100 :.2f}%")
        print('-' * 30, task_type, '-' * 30)
    acc_dict['Total Acc'] = f"{correct / total * 100 :.2f}%"
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as f:
        json.dump({
            "acc_dict": acc_dict,
            "res_list": res_list
        }, f)

    final_res = dict()
    correct = 0
    total = 0
    for k, v in acc_dict.items():
        if k == 'Total Acc':
            continue
        final_res[k] = int(v[0]) / int(v[1]) * 100
        correct += int(v[0])
        total += int(v[1])
    final_res['Avg'] = correct / total * 100

    print(final_res)

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
