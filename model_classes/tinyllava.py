from eval_model import EvalModel
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append("/content/drive/MyDrive/dev-bench/TinyLLaVA_Factory")
sys.path.append("./")
from TinyLLaVA_Factory.tinyllava.data.image_preprocess import ImagePreprocess
from model_classes.modeling_tinyllava_phi import tokenizer_image_token, conv_phi_v0, process_images

# class TinyLlavaEvalModel(EvalModel):
#     def __init__(self, model, tokenizer, device="cpu"):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model = model.to(self.device)
#         self.tokenizer = tokenizer
#         self.image_processor = ImagePreprocess(image_processor=model.vision_tower._image_processor, data_args=model.config)
#         self.get_similarity_scores = self.get_all_sim_scores
                                                                  
#     def get_all_sim_scores(self, dataloader, max_length=50, num_beams=5):
#         generated_texts = []

#         with torch.no_grad():
#             for idx, d in enumerate(tqdm(dataloader, desc="Processing data")):
#                 trial_texts = []
#                 for i, image in enumerate(d["images"]):
#                     if image.mode != 'RGB':
#                         image = image.convert('RGB')
                    
#                     # # Process the image
#                     # image_tensor = self.image_processor(image).unsqueeze(0).to(self.model.device, dtype=torch.float16)
#                     # print("Image tensor shape:", image_tensor.shape)

#                     for text in d['text']:
#                         prompt = f"Caption: {text}. Does the caption match the image? Answer either Yes or No." # Adjust the prompt for text generation
#                         output_text = self.model.chat(prompt=prompt, image=image, tokenizer=self.tokenizer)
#                         print("Output text:", output_text)
#                 #         inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

#                 #         with torch.inference_mode():
#                 #             generated_ids = self.model.generate(
#                 #                 input_ids=inputs['input_ids'],
#                 #                 attention_mask=inputs['attention_mask'],
#                 #                 images=image_tensor,
#                 #                 max_length=max_length,
#                 #                 num_beams=num_beams,
#                 #                 early_stopping=True
#                 #             )

#                 #         # Decode the generated tokens to text
#                 #         generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#                 #         print("Generated text:", generated_text)
#                 #         trial_texts.append(generated_text)

#                 # generated_texts.append(trial_texts)

#         return generated_texts





class TinyLlavaEvalModel(EvalModel):
    def __init__(self, model, tokenizer, device="cpu"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.image_processor = ImagePreprocess(image_processor=model.vision_tower._image_processor, data_args=model.config)
                                                                  
    def get_all_sim_scores(self, dataloader):
        all_sims = []
        image_tensor_temp = None

        with torch.no_grad():
            for idx, d in enumerate(tqdm(dataloader, desc="Processing data")):
                trial_sims = []
                for i, image in enumerate(d["images"]):
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    print(image)
                    
                    # Use CLIPImageProcessor to process the image
                    # image_tensor = self.image_processor(image).unsqueeze(0).to(self.model.device, dtype=torch.float16) 
                    image_tensor = process_images(image, self.model.vision_tower._image_processor, {}).to(self.device)
                    print("image tensor shape:", image_tensor.shape)

                    # # Flatten the tensor and print a portion of it
                    # flattened_tensor = image_tensor.flatten()
                    
                    # # Save the image tensor to a text file
                    # file_name = f"image_tensor_{idx}_{i}.txt"
                    # np.savetxt(file_name, flattened_tensor.cpu().numpy(), fmt='%.6f')
                    # print(f"Saved tensor to {file_name}")

                    for text in d['text']:
                        prompt = f"<image>\nCaption: {text}. Does the caption match the image? Answer either Yes or No only."
                        conv = conv_phi_v0.copy()
                        conv.append_message(conv.roles[0], prompt)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt()
                        # inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

                        input_ids = (
                            tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
                            .unsqueeze(0).to(self.device)
                        )

                        with torch.inference_mode():
                            outputs = self.model.generate(
                                # inputs=inputs['input_ids'],
                                inputs=input_ids,
                                # prompt=prompt,
                                # attention_mask=inputs['attention_mask'],
                                images=image_tensor,
                                # tokenizer=self.tokenizer,
                                max_new_tokens=10,
                                output_scores=True,
                                output_logits=True,
                                return_dict_in_generate=True,
                            )
                            logits = outputs.logits[0]
                            print(logits)
                            #print(logits.shape)
                            #logits.softmax(dim=-1)  #(28,51200)
                            #logits_maxed = logits.argmax(dim=-1) #(28,)
                            #print(logits_maxed)


                        yes_token_id = self.tokenizer.encode("Yes")[0]
                        no_token_id = self.tokenizer.encode("No")[0]

                        yes_logits = logits[:, yes_token_id]
                        no_logits = logits[:, no_token_id]

                        pair_logits = torch.stack((yes_logits, no_logits), dim=0).cpu().numpy()
                        trial_sims.append(pair_logits)
                        print("logits pair:", pair_logits)

                all_sims.append(np.array(trial_sims))

        return np.array(all_sims)
