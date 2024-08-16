from eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import sys
import torchvision.transforms as transforms
sys.path.append("/Users/sunnyyu/Documents/GitHub/dev-bench/TinyLLaVA_Factory")
from TinyLLaVA_Factory.tinyllava.data.image_preprocess import ImagePreprocess

class CogVlmEvalModel(EvalModel):
    def __init__(self, model, tokenizer, device="cpu"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.get_similarity_scores = self.get_all_sim_scores

    def get_all_sim_scores(self, dataloader):
        all_sims = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                trial_sims = []
                for image in d["images"]:
                    # Convert image to RGB if it's not already in that format
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    for text in d['text']:
                        prompt = f"Caption: {text}. Does the caption match the image? Answer either Yes or No."
                        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=prompt, images=[image])
                        inputs = {
                            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
                            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
                            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
                            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
                        }
                        logits = self.model(**inputs).logits.squeeze()
                        yes_token_id = self.processor.tokenizer.encode("Yes")[1]
                        no_token_id = self.processor.tokenizer.encode("No")[1]

                        yes_logits = logits[-1, yes_token_id]
                        no_logits = logits[-1, no_token_id]

                        pair_logits = torch.stack((yes_logits, no_logits), dim=0).cpu().numpy()
                        trial_sims.append(pair_logits)

                all_sims.append(np.array(trial_sims))

        return np.array(all_sims)



