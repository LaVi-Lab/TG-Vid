from torchvision import transforms
from torchvision.transforms import InterpolationMode

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
normalize = transforms.Normalize(mean, std)
type_transform = transforms.Lambda(lambda x: x.float().div(255.0))
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224,
            scale=(0.5, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        #transforms.RandomHorizontalFlip(),
        type_transform,
        normalize,
    ]
)

anno_root_it = '/path/to/Data-TG-Vid/Training'

# ============== pretraining datasets=================
available_corpus = dict(
    # VideoID of classification_k710_new: refer to k710/train_new.json
    classification_k710_new=[
        f"{anno_root_it}/video/classification/k710/train_new.json", 
        "",
        "video"
    ],
    classification_ssv2_new=[
        f"{anno_root_it}/video/classification/ssv2/train_new.json", 
        "ssd:s3://video_pub/ssv2_video",
        "video"
    ],
    reasoning_next_qa_new=[
        f"{anno_root_it}/video/reasoning/next_qa/train_new.json", 
        "p2:s3://nextqa",
        "video"
    ],
    reasoning_clevrer_qa_new=[
        f"{anno_root_it}/video/reasoning/clevrer_qa/train_new.json", 
        "p2:s3://clevrer/video_train",
        "video"
    ],
    reasoning_clevrer_mc_new=[
        f"{anno_root_it}/video/reasoning/clevrer_mc/train_new.json",  
        "p2:s3://clevrer/video_train",
        "video"
    ],
    
    # ST-LLM provides train_full_flat_fix.json
    # ST-LLM's conversation_videochatgpt = VideoChat2's conversation_videochatgpt
    caption_videochatgpt=[
        f"{anno_root_it}/video/conversation/videochatgpt/train_full_flat_fix.json", 
        "p2:s3://ANet/ANet_320p_fps30",
        "video"
    ],
    
    vqa_webvid_qa=[
        f"{anno_root_it}/video/vqa/webvid_qa/train.json", 
        "p2:s3://WebVid2M",
        "video"
    ],
)


