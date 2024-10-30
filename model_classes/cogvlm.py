from eval_model import GenEvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class CogVlmEvalModel(GenEvalModel):
    def __init__(self, model, tokenizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.model = model.to(self.device)
        self.processor = type("processor",(object,),dict(tokenizer=tokenizer))
        self.get_similarity_scores = self.get_all_sim_scores

    def get_ntp_logits(self, image, text):
        prompt = f"Caption: {text}. Does the caption match the image? Answer either Yes or No."
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[inputs['images'][0].to(self.device).to(torch.bfloat16)]],
        }
        logits = self.model(**inputs).logits.squeeze()
        return logits

    def get_ll_logits(self, image, text):
        prompt = f"Describe this image. {text}"
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(self.device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(self.device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(self.device),
            'images': [[inputs['images'][0].to(self.device).to(torch.bfloat16)]],
            'labels': inputs['input_ids'].unsqueeze(0).to(self.device)
        }
        outputs = self.model(**inputs)
        loglik = -outputs.loss
        return loglik
