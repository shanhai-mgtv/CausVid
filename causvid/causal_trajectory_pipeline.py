from causvid.models.model_interface import (
    InferencePipelineInterface,
    DiffusionModelInterface,
    TextEncoderInterface
)
from causvid.scheduler import SchedulerInterface
from typing import List
import torch
import torch.distributed as dist


class CausalInferenceWrapper(InferencePipelineInterface):
    def __init__(self, denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: DiffusionModelInterface, 
                 num_frame_per_block: int,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        self.num_frame_per_block = num_frame_per_block
        self.frame_seq_length = 44*80//4
        self.num_transformer_blocks = 30
        self.kv_cache = None
        self.crossattn_cache = None
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
    
    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache = []

        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, 21*44*80//4, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 21*44*80//4, 24, 128], dtype=dtype, device=device)
            })

        self.kv_cache = kv_cache  # always store the clean cache
    
    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 24, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache = crossattn_cache  # always store the clean cache

    def inference_with_trajectory(self, noise: torch.Tensor, conditional_dict: dict, use_init_noise: bool = True) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        
        ## chunk数
        num_blocks = num_frames // self.num_frame_per_block
        exit_flags = self.generate_and_sync_list(
            num_blocks=num_blocks, num_denoising_steps=len(self.denoising_step_list), device=noise.device
        )

        for block_index in range(num_blocks):
            noise_input = noise[:, block_index * self.num_frame_per_block:(block_index+1) * self.num_frame_per_block]
            init_noise = noise_input.clone()
            for index, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (index == exit_flags[block_index])
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], dtype=torch.long, device=noise.device) * current_timestep
                
                if not exit_flag:
                    with torch.no_grad():
                        pred_image_or_video = self.generator(
                            noisy_image_or_video=noise_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                            current_end=(block_index+1) * self.num_frame_per_block * self.frame_seq_length,
                            return_flow=False
                        )  # [B, F, C, H, W]
                        next_timestep = self.denoising_step_list[index + 1]
                    
                        noise_input = self.scheduler.add_noise(
                            pred_image_or_video.flatten(0, 1),
                            init_noise.flatten(0, 1) if use_init_noise else torch.randn_like(pred_image_or_video.flatten(0, 1)),
                            next_timestep * torch.ones([batch_size], device="cuda", dtype=torch.long),
                        ).unflatten(0, pred_image_or_video.shape[:2])
                else:
                    pred_image_or_video = self.generator(
                            noisy_image_or_video=noise_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                            current_end=(block_index+1) * self.num_frame_per_block * self.frame_seq_length,
                            return_flow=False
                        )  # [B, F, C, H, W]
                    break
            # get block index chunk 
            output[:, block_index * self.num_frame_per_block:(block_index+1) * self.num_frame_per_block] = pred_image_or_video
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=pred_image_or_video,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                    current_end=(block_index+1) * self.num_frame_per_block * self.frame_seq_length,
                    return_flow=False
                )  # [B, F, C, H, W]
        ## TODO: add noise 
        # if exit_flags[0] == len(self.denoising_step_list) - 1:
        #     denoised_timestep_to = 0
        #     denoised_timestep_from = 1000 - torch.argmin(
        #         (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        #  else:
        #     denoised_timestep_to = 1000 - torch.argmin(
        #         (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
        #     denoised_timestep_from = 1000 - torch.argmin(
        #         (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        return output

    def inference_with_trajectory_ode(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        output = torch.zeros(
            [batch_size, num_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )
        ## chunk数
        num_blocks = num_frames // self.num_frame_per_block
        exit_flags = self.generate_and_sync_list(
            num_blocks=num_blocks, num_denoising_steps=len(self.denoising_step_list), device=noise.device
        )

        for block_index in range(num_blocks):
            noise_input = noise[:, block_index * self.num_frame_per_block:(block_index+1) * self.num_frame_per_block]
            for index, current_timestep in enumerate(self.denoising_step_list):
                exit_flag = (index == exit_flags[block_index])
                timestep = torch.ones(
                    [batch_size, self.num_frame_per_block], dtype=torch.long, device=noise.device) * current_timestep
                
                if not exit_flag:
                    with torch.no_grad():
                        flow_pred, pred_image_or_video = self.generator(
                            noisy_image_or_video=noise_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                            current_end=(block_index+1) * self.num_frame_per_block * self.frame_seq_length,
                            return_flow=True
                        )  # [B, F, C, H, W]
                        next_timestep = self.denoising_step_list[index + 1]

                        noise_input = self.scheduler.add_noise_ode(
                            pred_image_or_video.flatten(0, 1),
                            noise_input.flatten(0, 1),
                            next_timestep * torch.ones([batch_size], device="cuda", dtype=torch.long),
                            flow_pred.flatten(0, 1),
                        ).unflatten(0, noise_input.shape[:2])
                else:
                    pred_image_or_video = self.generator(
                            noisy_image_or_video=noise_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                            current_end=(block_index+1) * self.num_frame_per_block * self.frame_seq_length,
                            return_flow=False
                        )  # [B, F, C, H, W]
                    break
            # get block index chunk 
            output[:, block_index * self.num_frame_per_block:(block_index+1) * self.num_frame_per_block] = pred_image_or_video
            self.generator(
                noisy_image_or_video=noise_input,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=block_index * self.num_frame_per_block * self.frame_seq_length,
                current_end=(block_index+1) * self.num_frame_per_block * self.frame_seq_length,
                return_flow=False
            )  # [B, F, C, H, W]
        ## TODO: add noise 
        # if exit_flags[0] == len(self.denoising_step_list) - 1:
        #     denoised_timestep_to = 0
        #     denoised_timestep_from = 1000 - torch.argmin(
        #         (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        #  else:
        #     denoised_timestep_to = 1000 - torch.argmin(
        #         (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
        #     denoised_timestep_from = 1000 - torch.argmin(
        #         (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        return output