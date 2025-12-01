from causvid.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from typing import List
import torch


class BidirectionalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()

        # Step 2: Initialize all bidirectional wan hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def inference(self, noise: torch.Tensor, text_prompts: List[str], neg_prompts: List[str], 
                  cfg_scale: float = 1.0) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        cfg = False
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        if cfg_scale > 1.0:
            neg_condition_dict = self.text_encoder(
                text_prompts=neg_prompts
            )
            cfg = True

        # initial point
        noisy_image_or_video = noise

        for index, current_timestep in enumerate(self.denoising_step_list):
            print(f"当前denoising步: {current_timestep}")
            cond_pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
            )  # [B, F, C, H, W]
            if cfg:
                uncond_pred_image_or_video = self.generator(
                    noisy_image_or_video=noisy_image_or_video,
                    conditional_dict=neg_condition_dict,
                    timestep=torch.ones(
                        noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
                )  # [B, F, C, H, W]
                pred_image_or_video = cond_pred_image_or_video + (
                    cond_pred_image_or_video - uncond_pred_image_or_video
                ) * cfg_scale
            else:
                pred_image_or_video = cond_pred_image_or_video

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device)

                noisy_image_or_video = self.scheduler.add_noise(
                    pred_image_or_video.flatten(0, 1),
                    noise.flatten(0, 1),
                    # torch.randn_like(pred_image_or_video.flatten(0, 1)),
                    next_timestep.flatten(0, 1)
                ).unflatten(0, noise.shape[:2])

        video = self.vae.decode_to_pixel(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video

    def inference_with_scheduler(self, noise: torch.Tensor, text_prompts: List[str], neg_prompts: List[str], 
                  cfg_scale: float = 1.0) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        cfg = False
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        if cfg_scale > 1.0:
            neg_condition_dict = self.text_encoder(
                text_prompts=neg_prompts
            )
            cfg = True

        # initial point
        latents = noise

        for index, current_timestep in enumerate(self.denoising_step_list):
            print(f"当前denoising步: {current_timestep}")
            current_timestep=torch.ones(
                    noise.shape[0], dtype=torch.long, device=noise.device) * current_timestep

            cond_flow_pred = self.generator.model(
                latents.permute(0, 2, 1, 3, 4),
                t=current_timestep, context=conditional_dict["prompt_embeds"],
                seq_len=self.generator.seq_len
            ).permute(0, 2, 1, 3, 4)

            if cfg:
                uncond_flow_pred = self.generator.model(
                    latents.permute(0, 2, 1, 3, 4),
                    t=current_timestep, context=neg_condition_dict["prompt_embeds"],
                    seq_len=self.generator.seq_len
                ).permute(0, 2, 1, 3, 4)

                flow_pred = cond_flow_pred + cfg_scale * (
                    cond_flow_pred - uncond_flow_pred
                )
            else:
                flow_pred = cond_flow_pred
            
            # latents = self.scheduler.step(flow_pred, current_timestep, latents).to(dtype=noise.dtype)
            x0_pred = self.generator._convert_flow_pred_to_x0(
                flow_pred,
                xt=latents,
                timestep=current_timestep
            )
            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                    noise.shape[0], dtype=torch.long, device=noise.device)

                latents = self.scheduler.add_noise_ode(
                    x0_pred,
                    # noise,
                    torch.randn_like(latents),
                    next_timestep,
                    flow_pred
                )

        # video = self.vae.decode_to_pixel(latents)
        video = self.vae.decode_to_pixel(x0_pred)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video