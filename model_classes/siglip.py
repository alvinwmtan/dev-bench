from eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

class SiglipEvalModel(EvalModel):
    def __init__(self, model, processor=None, tokenizer=None, model_embed=None, processor_embed = None, device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_embed = model_embed.to(self.device)
        self.processor_embed = processor_embed.to(self.device)

        self.get_similarity_scores = lambda **x: torch.sigmoid(self.model(**x).logits_per_image)

    def get_all_image_feats(self, dataloader):
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                inputs = self.processor_embed(images=d['images'], return_tensors="pt")
                image_features = self.model_embed.get_image_features(**inputs).detach().numpy()
                all_feats.append(image_features)
        return np.concatenate(all_feats, axis=0)

    def get_all_text_feats(self, dataloader):
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                inputs = self.tokenizer(d['text'], padding = "max_length", return_tensors = "pt")
                text_features = self.model_embed.get_text_features(**inputs).detach().numpy()
        return text_features