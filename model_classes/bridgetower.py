from eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

class BridgetowerEvalModel(EvalModel):
    def __init__(self, model, image_model=None, image_processor = None, processor=None, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.processor = processor
        self.image_processor = image_processor
        self.image_model = image_model

    def get_all_image_feats(self, dataloader):
        """
        Gets image embeddings from a dataloader using a model that outputs embeddings.
        
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - model: the model used to extract image embeddings
        
        Outputs:
        - a numpy array of shape [num_images, embed_dim]
        """
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                for image in d["images"]:
                    inputs = self.image_processor(images=[image], text=[""], 
                                            return_tensors="pt", padding=True, truncation=True)
                    outputs = self.image_model(**inputs)

                    image_features = outputs.image_features 
                    pooled_feats = image_features.mean(dim=1).squeeze().detach().numpy()
                    if len(pooled_feats.shape) == 1: 
                        all_feats.append(pooled_feats)
                    elif len(pooled_feats.shape) == 2: 
                        all_feats.extend(pooled_feats)
                    else:
                        print(f"Unexpected shape of pooled features: {pooled_feats.shape}")
        
        return np.array(all_feats)

    def get_all_text_feats(self, dataloader):
        """
        Gets text features from a dataloader using a model that outputs logits
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the appropriate input processor for the model
        - model: the model used to extract text features
        Outputs:
        - a numpy array of shape [num_texts, embed_dim]
        """
        blank_image = Image.new('RGB', (224, 224), (0, 0, 0))

        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                inputs = self.processor(images=blank_image, text=d["text"], 
                                        return_tensors="pt", padding=True, 
                                        truncation=True, max_length=512)
                outputs = self.model(**inputs)
                feats = outputs.logits[:, 0].detach().numpy()
                all_feats.append(feats)
        return np.concatenate(all_feats, axis=0)

    def get_all_sim_scores(self, dataloader):
        """
        Gets image--text similarity scores from a dataloader using Bridge Tower model
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - processor: the BridgeTowerProcessor
        - model: the BridgeTowerModel
        Outputs:
        - a numpy array of shape [num_trials, num_images_per_trial, num_texts_per_trial]
        """
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                num_images = len(d["images"])
                num_texts = len(d["text"])
                sims = np.zeros((num_images, num_texts))

                for i, image in enumerate(d["images"]):
                    for j, text in enumerate(d["text"]):
                        inputs = self.processor(images=image, text=text, return_tensors="pt", 
                                           padding=True, truncation=True)
                        outputs = self.model(**inputs)
                        sims[i, j] = outputs.logits[0, 1].item()

                all_sims.append(sims)
        return np.stack(all_sims, axis=0)