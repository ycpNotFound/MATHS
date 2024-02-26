import sys
sys.path.append('./')


from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_NAMES_MAPPING
from .model.bertv2 import (
    BertConfigV2, 
    BertLMHeadModelV2
)
from .model.llama_cross_attn import (
    LlamaForCrossAttention,
    LlamaCrossAttnConfig
)
from .model.vision_encoder_decoder import (
    myVisionEncoderDecoderConfig,
    myVisionEncoderDecoderModel
)
CONFIG_MAPPING.register('bertv2', BertConfigV2)
CONFIG_MAPPING.update([("bertv2", BertConfigV2)])
MODEL_NAMES_MAPPING["bertv2"] = "bertv2"
MODEL_FOR_CAUSAL_LM_MAPPING.register(BertConfigV2, BertLMHeadModelV2)
MODEL_NAMES_MAPPING.update([("bertv2", "BERTV2")])

CONFIG_MAPPING.register('llama-cross-attn', LlamaCrossAttnConfig)
CONFIG_MAPPING.update([("llama-cross-attn", LlamaCrossAttnConfig)])
MODEL_NAMES_MAPPING["llama-cross-attn"] = "llama-cross-attn"
MODEL_FOR_CAUSAL_LM_MAPPING.register(LlamaCrossAttnConfig, LlamaForCrossAttention)
MODEL_NAMES_MAPPING.update([("llama-cross-attn", "llama-cross-attn")])

# CONFIG_MAPPING.register('my-vision-encoder-decoder', myVisionEncoderDecoderConfig)
# CONFIG_MAPPING.update([('my-vision-encoder-decoder', myVisionEncoderDecoderConfig)])
# MODEL_NAMES_MAPPING['my-vision-encoder-decoder'] = 'my-vision-encoder-decoder'
# MODEL_FOR_CAUSAL_LM_MAPPING.register(LlamaCrossAttnConfig, LlamaForCrossAttention)
# MODEL_NAMES_MAPPING.update([("llama-cross-attn", "llama-cross-attn")])