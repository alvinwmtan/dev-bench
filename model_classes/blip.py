from eval_model import EvalModel
from tqdm import tqdm
import torch

class BlipEvalModel(EvalModel):
    def __init__(self, model, processor=None, image_model=None, device="cpu"):
        self.device = torch.device(device)  
        self.model = model.to(self.device)  
        self.processor = processor
        self.image_model = image_model

        self.get_image_features = self.image_model.get_image_features
        self.get_textfeatures = self.image_model.get_text_features
        self.get_similarity_scores = lambda **x: self.model(**x).itm_score
