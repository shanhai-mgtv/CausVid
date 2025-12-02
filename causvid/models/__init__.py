from .wan.wan_wrapper import WanTextEncoder, WanVAEWrapper, WanDiffusionWrapper, CausalWanDiffusionWrapper
from .wan22.wan_wrapper import Wan22TextEncoder, Wan22VAEWrapper, Wan22DiffusionWrapper
from causvid.bidirectional_trajectory_pipeline import BidirectionalInferenceWrapper
from .sdxl.sdxl_wrapper import SDXLWrapper, SDXLTextEncoder, SDXLVAE
from transformers.models.t5.modeling_t5 import T5Block


DIFFUSION_NAME_TO_CLASS = {
    "sdxl": SDXLWrapper,
    "wan": WanDiffusionWrapper,
    "wan22": Wan22DiffusionWrapper,
    "causal_wan": CausalWanDiffusionWrapper
}


def get_diffusion_wrapper(model_name):
    return DIFFUSION_NAME_TO_CLASS[model_name]


TEXTENCODER_NAME_TO_CLASS = {
    "sdxl": SDXLTextEncoder,
    "wan": WanTextEncoder,
    "wan22": Wan22TextEncoder,
    "causal_wan": WanTextEncoder
}


def get_text_encoder_wrapper(model_name):
    return TEXTENCODER_NAME_TO_CLASS[model_name]


VAE_NAME_TO_CLASS = {
    "sdxl": SDXLVAE,
    "wan": WanVAEWrapper,
    "wan22": Wan22VAEWrapper,
    "causal_wan": WanVAEWrapper   # TODO: Change the VAE to the causal version
}


def get_vae_wrapper(model_name):
    return VAE_NAME_TO_CLASS[model_name]


PIPELINE_NAME_TO_CLASS = {
    "sdxl": BidirectionalInferenceWrapper,
    "wan": BidirectionalInferenceWrapper,
    "wan22": BidirectionalInferenceWrapper,
}


def get_inference_pipeline_wrapper(model_name, **kwargs):
    return PIPELINE_NAME_TO_CLASS[model_name](**kwargs)


BLOCK_NAME_TO_BLOCK_CLASS = {
    "T5Block": T5Block
}


def get_block_class(model_name):
    return BLOCK_NAME_TO_BLOCK_CLASS[model_name]
