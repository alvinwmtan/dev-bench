from eval_model import GenEvalModel
from tqdm import tqdm
import re
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class LlavaEvalModel(GenEvalModel):
    def get_ntp_logits(self, image, text):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(image).to(self.device)

        prompt = f"[INST] <image>\nCaption: {text}. Does the caption match the image? Answer either Yes or No. [/INST]"
        inputs = self.processor(text=prompt, images=image_tensor, return_tensors='pt').to(self.device)
        logits = self.model(**inputs).logits.squeeze()
        return logits

    def get_ll_logits(self, image, text):
        prompt = f"[INST] <image>\nDescribe this image. [/INST] {text}"
        inputs = self.processor(text=prompt, images=image_tensor, return_tensors='pt').to(self.device) 
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        loglik = -outputs.loss
        return loglik