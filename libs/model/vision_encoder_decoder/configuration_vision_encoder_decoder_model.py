from transformers import VisionEncoderDecoderConfig

class myVisionEncoderDecoderConfig(VisionEncoderDecoderConfig):
    model_type = "vision-encoder-decoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    