from eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import importlib.util
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from moondream.configuration_moondream import MoondreamConfig

class MoondreamEvalModel(EvalModel):
    def __init__(self, model, tokenizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        spec = importlib.util.spec_from_file_location("modified_moondream", "moondream/moondream.py") # path to moondream.py
        modified_moondream = importlib.util.module_from_spec(spec)
        sys.modules["modified_moondream"] = modified_moondream
        spec.loader.exec_module(modified_moondream)
        ModifiedMoondreamModel = modified_moondream.Moondream

        config = MoondreamConfig()
        model = ModifiedMoondreamModel(config)
        model.load_state_dict(model.state_dict())

        self.device = device
        self.model = model.to(self.device)
        self.processor = type("processor",(object,),dict(tokenizer=tokenizer))

    def get_ntp_logits(self, image, text):
        enc_image = self.model.encode_image(image_path)
        prompt = f"<image> Caption: {text}. Does the caption match the image? Answer either Yes or No."
        inputs_embeds = self.model.input_embeds(prompt, enc_image, self.processor.tokenizer)

        attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long)
        outputs = self.model.text_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
        )
        
        logits = outputs.logits[:, -1, :]
    
    def get_ll_logits(self, image, text):
        prompt = f"<image> Describe this image. {text}"
        inputs_embeds = self.model.input_embeds(prompt, enc_image, self.processor.tokenizer)
        input_ids = self.processor.tokenizer(prompt.replace("<image> ", ""), return_tensors="pt", add_special_tokens=False).input_ids
        input_ids = torch.cat([(torch.ones((1, 730), dtype=torch.int)*-100), input_ids], dim=1) # prepend BOS and image embeds
        outputs = self.model.text_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            labels=input_ids.to(self.device)
        )
    
        loglik = -outputs.loss
        return loglik

    def get_all_sim_scores(self, dataloader):
        all_sims = []
        with torch.no_grad():
            for d in dataloader:
                trial_sims = []
                for image_path in d["images"]:
                    for text in d['text']:
                        logits = get_ntp_logits(image, text)
                        
                        yes_token_id = self.processor.tokenizer(" Yes", add_special_tokens=False).input_ids[0]
                        no_token_id = self.processor.tokenizer(" No", add_special_tokens=False).input_ids[0]
                        yes_logits = logits[:,yes_token_id].squeeze()
                        no_logits = logits[:, no_token_id].squeeze()
                        pair_logits = torch.stack((yes_logits, no_logits), dim=0).cpu().numpy()
     
                        trial_sims.append(pair_logits)
                all_sims.append(np.array(trial_sims))
        return np.array(all_sims)