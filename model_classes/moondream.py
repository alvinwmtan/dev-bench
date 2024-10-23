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
    def __init__(self, model, tokenizer, processor=None, device="cpu"):
        spec = importlib.util.spec_from_file_location("modified_moondream", "/dev-bench/moondream/moondream.py") # path to moondream.py
        modified_moondream = importlib.util.module_from_spec(spec)
        sys.modules["modified_moondream"] = modified_moondream
        spec.loader.exec_module(modified_moondream)
        ModifiedMoondreamModel = modified_moondream.Moondream
        # model_id = "vikhyatk/moondream2"
        # revision = "2024-05-20"
        # pretrained_model = AutoModelForCausalLM.from_pretrained(
        #     model_id, trust_remote_code=True, revision=revision
        # )
        # tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        config = MoondreamConfig()
        # Create an instance of your modified model
        model = ModifiedMoondreamModel(config)
        print("model loaded successfully")

        # Load the state dict from the pretrained model
        model.load_state_dict(model.state_dict())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    def get_all_sim_scores(self, dataloader):
        """
        Gets image--text similarity scores from a dataloader using a specific prompt format.
        -----
        Inputs:
        - dataloader: a dataloader constructed from a DevBenchDataset
        - model: the model used to obtain similarity scores
        - tokenizer: the tokenizer used to prepare text inputs
        Outputs:
        - a numpy array of shape [num_trials, num_images * num_texts, 2] where the last dimension contains logits for "yes" and "no"
        """
        all_sims = []
        with torch.no_grad():
            for d in dataloader:
                trial_sims = []
                for image_path in d["images"]:
                    enc_image = self.model.encode_image(image_path)
                    #print(enc_image)
                    for text in d['text']:
                        prompt = f"<image> Caption: {text}. Does the caption match the image? Answer either Yes or No."
                        #print(text)
                        inputs_embeds = self.model.input_embeds(prompt, enc_image, self.tokenizer)
                        #print(inputs_embeds)

                        # Generate with explicit attention mask and pad token settings
                        attention_mask = torch.ones(inputs_embeds.shape[:-1], dtype=torch.long)  # Assuming inputs_embeds is 2D
                        outputs = self.model.text_model(
                            inputs_embeds=inputs_embeds,
                            #attention_mask=attention_mask,
                            #pad_token_id=tokenizer.eos_token_id,
                            output_hidden_states=True,
                            #return_dict_in_generate=True,
                            #max_new_tokens=1,  # Adjust as necessary to generate enough tokens
                        )
                    
                        
                        logits = outputs.logits[:, -1, :]

                        # Use the exact tokens as they appear in the tokenizer's vocabulary
                        yes_token_id = self.tokenizer(" Yes", add_special_tokens=False).input_ids[0]
                        no_token_id = self.tokenizer(" No", add_special_tokens=False).input_ids[0]

                        # Pass each hidden state through the final linear layer to get logits
                        #logits = model.text_model.lm_head(hidden_states)

                        # Get logits for "yes" and "no"
                        yes_logits = logits[:,yes_token_id].squeeze()
                        no_logits = logits[:, no_token_id].squeeze()

            

                        # Stack the logits for "yes" and "no" to form the required output shape
                        pair_logits = torch.stack((yes_logits, no_logits), dim=0).cpu().numpy()
                        #print(enc_image, text, pair_logits)
     
                        trial_sims.append(pair_logits)
                

                all_sims.append(np.array(trial_sims))


        return np.array(all_sims)
    
    def get_all_image_feats(self, dataloader):
        all_feats = []

        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing batches"):
                # Assuming the model expects tensors on the device
                for image_path in d["images"]:
                    enc_image = self.model.encode_image(image_path)
                    
                    # Ensure the encoded image tensor is moved to the CPU for further processing
                    enc_image = enc_image.to('cpu')

                    if len(enc_image.shape) > 2:
                        # If enc_image has more than 2 dimensions, apply mean pooling
                        enc_image = enc_image.mean(dim=1)
                    all_feats.append(enc_image.numpy())  # Convert tensor to numpy array
                    print(enc_image.shape)

        return np.concatenate(all_feats, axis=0)

    def get_all_text_feats(self,dataloader):
        all_feats = []
        with torch.no_grad():
            for d in tqdm(dataloader, desc="Processing data"):
                for text in d['text']:
                    inputs_embeds = self.model.input_embeds(text, None, self.tokenizer)
                    if len(inputs_embeds.shape) > 2:
                        inputs_embeds = inputs_embeds.mean(dim=1)
                    all_feats.append(inputs_embeds)
                    print(inputs_embeds.shape)
                    print(inputs_embeds)
        return np.concatenate(all_feats, axis=0)