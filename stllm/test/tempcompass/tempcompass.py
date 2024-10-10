import os 
import json
import math
import numpy as np
import cv2
import io
import imageio
from mmengine.fileio import FileClient
client = FileClient('disk')
from decord import VideoReader, cpu
from PIL import Image
import torchvision.transforms as T
from stllm.test.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset
import torch

from stllm.conversation.mvbench_conversation import ask, answer, EasyDict


data_list = {
    "TempCompass-Multi-Choice-QA-Action": ("multi-choice/TempCompass-Multi-Choice-QA-Action.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Multi-Choice-QA-Direction": ("multi-choice/TempCompass-Multi-Choice-QA-Direction.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Multi-Choice-QA-Speed": ("multi-choice/TempCompass-Multi-Choice-QA-Speed.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Multi-Choice-QA-Event-Order": ("multi-choice/TempCompass-Multi-Choice-QA-Event-Order.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Multi-Choice-QA-Attribute-Change": ("multi-choice/TempCompass-Multi-Choice-QA-Attribute-Change.json", "/path/to/TempCompass/videos/", "video", False),
    
    "TempCompass-Yes-No-QA-Action": ("yes_no/TempCompass-Yes-No-QA-Action.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Yes-No-QA-Direction": ("yes_no/TempCompass-Yes-No-QA-Direction.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Yes-No-QA-Speed": ("yes_no/TempCompass-Yes-No-QA-Speed.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Yes-No-QA-Event-Order": ("yes_no/TempCompass-Yes-No-QA-Event-Order.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Yes-No-QA-Attribute-Change": ("yes_no/TempCompass-Yes-No-QA-Attribute-Change.json", "/path/to/TempCompass/videos/", "video", False),

    "TempCompass-Caption-Matching-QA-Action": ("caption_matching/TempCompass-Caption-Matching-QA-Action.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Caption-Matching-QA-Direction": ("caption_matching/TempCompass-Caption-Matching-QA-Direction.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Caption-Matching-QA-Speed": ("caption_matching/TempCompass-Caption-Matching-QA-Speed.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Caption-Matching-QA-Event-Order": ("caption_matching/TempCompass-Caption-Matching-QA-Event-Order.json", "/path/to/TempCompass/videos/", "video", False),
    "TempCompass-Caption-Matching-QA-Attribute-Change": ("caption_matching/TempCompass-Caption-Matching-QA-Attribute-Change.json", "/path/to/TempCompass/videos/", "video", False),
}

data_dir = "/path/to/Data-TG-Vid/Testing/TempCompass"

class TempCompass_dataset(Dataset):
    def __init__(self, data_dir, data_list=data_list, num_segments=8, resolution=224, specified_item=None):
        self.data_list = []
        print(f"data_dir = {data_dir}")
        print(f"data_list = {data_list}")
        if specified_item:
            data_list = {specified_item: data_list[specified_item]}
        for k, v in data_list.items():
            print(f"k, v = {k}, {v}")
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)

        if bound:
            video_len = bound[1] - bound[0]
        else:
            video_len = max_frame / fps

        if self.num_segments > 0:
            num_segments = self.num_segments  
        else:  #fps 1
            if video_len < 4:
                num_segments = 4
            elif video_len > 16:
                num_segments = 16
            else:
                num_segments = math.floor(video_len)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        video_bytes = client.get(video_path)
        vr = VideoReader(io.BytesIO(video_bytes), ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)

        return torch_imgs
    
    def read_gif(self, video_path, bound=None, fps=25):
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
        max_frame = len(gif) - 1
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        if os.path.exists(video_path):
            max_frame = len(os.listdir(video_path))
        else:
            max_frame = len([k for k in client.list(video_path)])
            
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img_bytes = client.get(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            img = Image.open(io.BytesIO(img_bytes))
            images_group.append(img)
        torch_imgs = self.transform(images_group)

        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'video_path': video_path,
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type'],
        }

def get_residual_index(sample_segments, total_segments, devices):
    seg_size = float(total_segments) / sample_segments
    frame_indices = np.array([
    int((seg_size / 2) + np.round(seg_size * idx))
    for idx in range(sample_segments)
    ])
    frame_indices = torch.from_numpy(frame_indices).to(devices)
    return frame_indices

def infer_TempCompass(
        model,
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_llm=False,
        all_token=False,
        ask_simple=False,
        use_16frames_padding=False,
        use_32frames_padding=False,
    ):
    print(f"data_sample.keys = {data_sample.keys()}")
    video = data_sample["video"]
    TC, H, W = video.shape
    video = video.reshape(TC//3, 3, H, W).to("cuda:0")
    print(f"video.shape = {video.shape}")
    
    if use_16frames_padding:
        num_frames = 16
        padding_frames = num_frames - video.shape[0]
        if padding_frames > 0: # padded with last frame
            last_frame = video[-1].unsqueeze(0).repeat(padding_frames, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
            video = video.to("cuda:0")
            print(f"After padding, video.shape = {video.shape}")
    elif use_32frames_padding:
        num_frames = 32
        padding_frames = num_frames - video.shape[0]
        if padding_frames > 0: # padded with last frame
            last_frame = video[-1].unsqueeze(0).repeat(padding_frames, 1, 1, 1)
            video = torch.cat([video, last_frame], dim=0)
            video = video.to("cuda:0")
            print(f"After padding, video.shape = {video.shape}")
            
    video_list = []
    with torch.no_grad():
        if hasattr(model.model,'stllm_model'):
            encode_model = model.model.stllm_model
        else:
            encode_model = model.model.model.stllm_model
            
        video_emb, _, _ = encode_model.encode_img(video, data_sample['question'])
        
    if not all_token:
        video_emb = video_emb.mean(dim=0, keepdim=True)
    else:
        print(f"video_emb.shape = {video_emb.shape}")
        # video_emb = video_emb.view(1, -1, video_emb.size(-1))
        video_emb = video_emb.contiguous().view(1, -1, video_emb.size(-1))
    video_list.append(video_emb)

    chat = EasyDict({
        "system": system,
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video>\n"])
    
    if system_llm:
        prompt = system + data_sample['question'] + question_prompt
    else:
        prompt = data_sample['question'] + question_prompt
    
    ask(prompt, chat)

    llm_message = answer(
        conv=chat, model=model, ask_simple=ask_simple, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(f"chat = {chat}")
    print(f"Predict: {llm_message}")
    print(f"GT: {data_sample['answer']}")
    return llm_message

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

if __name__ == "__main__":
    dataset = TempCompass_dataset(data_dir, data_list, num_segments=16, resolution=224)