from eval_model import GenEvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

import sys
sys.path.append("/data/tanawm/dev-bench/TinyLLaVA_Factory")
sys.path.append("./")
from TinyLLaVA_Factory.tinyllava.data.image_preprocess import ImagePreprocess
from model_classes.modeling_tinyllava_phi import tokenizer_image_token, conv_phi_v0, process_images

class TinyLlavaEvalModel(GenEvalModel):
    def __init__(self, model, tokenizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.model = model.to(self.device)
        self.processor = type("processor",(object,),dict(tokenizer=tokenizer))
        self.image_processor = ImagePreprocess(image_processor=model.vision_tower._image_processor, data_args=model.config)
    
    def get_ntp_logits(self, image, text):
        image_tensor = process_images(image, self.model.vision_tower._image_processor, {}).to(self.device)
        prompt = f"<image>\nCaption: {text}. Does the caption match the image? Answer either Yes or No only."
        conv = conv_phi_v0.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.processor.tokenizer, -200, return_tensors="pt")
            .unsqueeze(0).to(self.device)
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                inputs=input_ids,
                images=image_tensor,
                max_new_tokens=10,
                output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )
            logits = outputs.logits[0]
        return logits
    
    def get_ll_logits(self, image, text):
        image_tensor = process_images(image, self.model.vision_tower._image_processor, {}).to(self.device)
        prompt = "A chat between a curious user and an artificial intelligence assistant. "+\
                    "The assistant gives helpful, detailed, and polite answers to the user's questions. "+\
                    f"USER: <image>\nDescribe this image. ASSISTANT: {text}"

        input_ids = (
            tokenizer_image_token(prompt, self.processor.tokenizer, -200, return_tensors="pt")
            .unsqueeze(0).to(self.device)
        )

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                images=image_tensor,
                labels=input_ids,
                return_dict=True,
            )
            loglik = -outputs.loss
        
        return loglik
