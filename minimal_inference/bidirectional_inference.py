import sys
sys.path.append("/mnt/cfs/shanhai/jyutong/hyc/causvid")
from causvid.models.wan22.bidirectional_inference import BidirectionalInferencePipeline
from huggingface_hub import hf_hub_download
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str, default=None)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)

args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

print("Load pipeline")
pipe = BidirectionalInferencePipeline(config, device="cuda")
print("finish load pipeline")

if args.checkpoint_folder:
    state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")['generator']
    pipe.generator.load_state_dict(state_dict)
    cfg_scale = 1.0
else:
    pipe.scheduler.set_timesteps(num_inference_steps=50)
    cfg_scale = 3.5
    pipe.denoising_step_list = pipe.scheduler.timesteps
    # pipe.denoising_step_list = [1000, 750, 500, 250]
    # cfg_scale = 3.5
    
pipe = pipe.to(device="cuda", dtype=torch.bfloat16)

print("load dataset")
dataset = TextDataset(args.prompt_file_path)
print("finish load dataset")

os.makedirs(args.output_folder, exist_ok=True)

print("start inference")
for index in tqdm(range(len(dataset))):
    prompt = dataset[index]
    video = pipe.inference(
        noise=torch.randn(
            1, 20, 48, 28, 52, generator=torch.Generator(device="cuda").manual_seed(1024),
            dtype=torch.bfloat16, device="cuda"
        ),
        text_prompts=[prompt],
        neg_prompts=[config.negative_prompt],
        cfg_scale=cfg_scale
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{index:03d}.mp4"), fps=16)
