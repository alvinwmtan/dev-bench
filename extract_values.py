import numpy as np
import torch
from tqdm import tqdm

def get_all_image_feats(dataloader, processor, get_image_features):
    """
    Gets image features from a dataloader
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - get_image_features: the model or model attribute used to
      extract image features
    Outputs:
    - a numpy array of shape [num_images, embed_dim]
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            inputs = processor(images=d["images"], return_tensors="pt")
            feats = get_image_features(**inputs).detach().numpy()
            all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)

def get_all_text_feats(dataloader, processor, get_text_features):
    """
    Gets text features from a dataloader
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - get_text_features: the model or model attribute used to
      extract text features
    Outputs:
    - a numpy array of shape [num_texts, embed_dim]
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            inputs = processor(text=d["text"], return_tensors="pt", 
                              padding=True)
            feats = get_text_features(**inputs).detach().numpy()
            all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)



def get_all_text_feats_flava(dataloader, processor, model):
    """
    Gets text features from a dataloader using the FLAVA model
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - model: the FLAVA model

    Outputs:
    - a list of numpy arrays, where each array represents the text features for a data point
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            inputs = processor(text=d["text"], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            feats = outputs.text_embeddings.detach().cpu().numpy()
            all_feats.extend(feats)
    return all_feats

def get_all_sim_scores(dataloader, processor, get_similarity_scores):
    """
    Gets image--text similarity scores from a dataloader
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - get_similarity_scores: the model or model attribute used to
      obtain similarity scores
    Outputs:
    - a numpy array of shape [num_trials, num_images_per_trial, 
      num_texts_per_trial]
    """
    all_sims = []
    with torch.no_grad():
        for d in dataloader:
            inputs = processor(images=d["images"], text=d["text"], 
                              return_tensors="pt", padding=True)
            sims = get_similarity_scores(**inputs).detach().numpy()
            all_sims.append(sims)
    return np.stack(all_sims, axis=0)

def get_all_sim_scores_gpu(dataloader, processor, get_similarity_scores):
    """
    Gets image--text similarity scores from a dataloader
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - get_similarity_scores: the model or model attribute used to
      obtain similarity scores
    Outputs:
    - a numpy array of shape [num_trials, num_images_per_trial, 
      num_texts_per_trial]
    """
    all_sims = []
    with torch.no_grad():
        for d in tqdm(dataloader, desc="Processing data"):
            images_tensor = torch.stack(d["images"]).to('cuda')
            inputs = processor(images=images_tensor, text=d["text"], return_tensors="pt", max_length=77, padding=True, return_codebook_pixles=True, return_image_mask=True)
            sims = get_similarity_scores(**inputs).detach().numpy()
            all_sims.append(sims)
    return np.stack(all_sims, axis=0)
  
  
def get_all_sim_scores_flava(dataloader, processor, model):
    all_sims = []
    with torch.no_grad():
        for data in dataloader:
            # Assume that processor and model can handle batches
            images_rgb = [image.convert("RGB") for image in data["images"]]
            inputs = processor(text=data["text"], images=images_rgb, return_tensors="pt", max_length=77, padding=True, return_codebook_pixels=True, return_image_mask=True)
            outputs = model(**inputs)
            scores = outputs.contrastive_logits_per_image.view(len(data["images"]), len(data["text"])).numpy()
            all_sims.append(scores)
    return np.array(all_sims)
  
def get_all_text_feats_blip(data_loader, processor, model):
    all_text_feats = []
    
    for batch in tqdm(data_loader, desc="Extracting text features"):
        text_features = model(batch["text"])
        all_text_feats.append(text_features)
    
    all_text_feats = torch.cat(all_text_feats, dim=0)
    return all_text_feats
  
def get_all_sim_scores_flava_aro(dataloader, processor, model):
    all_sims = []
    with torch.no_grad():
        for data in dataloader:
            for i in range(len(data["images"])):
                image_rgb = data["images"][i].convert("RGB")
                corresponding_texts = [data["text"][2*i], data["text"][2*i+1]]
                # Flatten the text inputs into a single batch
                inputs = processor(text=corresponding_texts, images=[image_rgb, image_rgb], return_tensors="pt", max_length=77, padding=True, return_codebook_pixels=True, return_image_mask=True)
                
                outputs = model(**inputs)
                scores = outputs.contrastive_logits_per_image.view(1, -1).numpy()  # Adjust view to handle multiple texts
                all_sims.append(scores)
    return np.array(all_sims)


import torch
import numpy as np

def get_all_sim_scores_bridgetower(dataloader, processor, model):
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
        for d in dataloader:
            # Prepare inputs with padding and truncation
            # Assuming each data point in the dataloader has multiple images and texts
            num_images = len(d["images"])
            num_texts = len(d["text"])
            sims = np.zeros((num_images, num_texts))

            for i, image in enumerate(d["images"]):
                for j, text in enumerate(d["text"]):
                    # Prepare inputs for each image-text pair
                    inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True)
                    # Forward pass
                    outputs = model(**inputs)
                    sims[i, j] = outputs.logits[0, 1].item()

            # Append the similarity scores for this batch to all_sims
            all_sims.append(sims)
    
    return np.stack(all_sims, axis=0)


from PIL import Image

def get_all_text_feats_bridgewater(dataloader, processor, model):
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
    # Create a blank (black) image with RGB channels and size 224x224
    blank_image = Image.new('RGB', (224, 224), (0, 0, 0))

    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            # Use a blank image for each text input
            inputs = processor(images=blank_image, text=d["text"], return_tensors="pt", padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            # Assuming the relevant text features are in the first position of logits
            feats = outputs.logits[:, 0].detach().numpy()  # Adjust indexing if necessary
            all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)


def get_all_sim_scores_cvcl(dataloader, preprocess, model):
    """
    Gets image--text similarity scores from a dataloader using the MultiModalLitModel
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - model: the MultiModalLitModel
    Outputs:
    - a numpy array of shape [num_trials, num_images_per_trial, num_texts_per_trial]
    """
    all_sims = []
    with torch.no_grad():
        for d in dataloader:
            # Process images
            images = [preprocess(img) for img in d["images"]]
            images = torch.stack(images).to(model.device)
            image_features = model.encode_image(images)

            # Tokenize and encode texts
            texts, texts_len = model.tokenize(d["text"])
            texts, texts_len = texts.to(model.device), texts_len.to(model.device)
            text_features = model.encode_text(texts, texts_len)

            # Get logits for image-text pairs
            logits_per_image, logits_per_text = model(images, texts, texts_len)
            sims = logits_per_image.detach().cpu().numpy()
            all_sims.append(sims)
    return np.stack(all_sims, axis=0)

def get_all_sim_scores_vilt(dataloader, processor, model):
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
        for d in dataloader:
            # Prepare inputs with padding and truncation
            # Assuming each data point in the dataloader has multiple images and texts
            num_images = len(d["images"])
            num_texts = len(d["text"])
            sims = np.zeros((num_images, num_texts))

            for i, image in enumerate(d["images"]):
                #print(image)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                #print(image.mode)
                scores = {}
                for j, text in enumerate(d["text"]):
                    # Prepare inputs for each image-text pair
                    #inputs = processor(images=image, text=text, return_tensors="pt", padding=True, truncation=True)
                    encoding = processor(image, text, return_tensors="pt")
                    # Forward pass
                    outputs = model(**encoding)
                    scores[text] = outputs.logits[0, :].item()
                    sims[i, j] = scores[text]

            # Append the similarity scores for this batch to all_sims
            all_sims.append(sims)
    
    return np.stack(all_sims, axis=0)

from tqdm import tqdm 

from PIL import Image

import torch
import numpy as np

def get_all_image_feats_bridgewater(dataloader, processor, model):
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
        for d in dataloader:
            # Process each image individually
            for image in d["images"]:
                # Convert image to RGB if not already in that format
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Pass a blank text to satisfy model requirements
                inputs = processor(images=[image], text=[""], return_tensors="pt", padding=True, truncation=True)
                
                # Model inference
                outputs = model(**inputs)
                
                # Extract image features
                image_features = outputs.image_features  # Shape: (batch_size, image_sequence_length, hidden_size)
                
                # Average pooling over the sequence length dimension to get (batch_size, hidden_size)
                pooled_feats = image_features.mean(dim=1).squeeze().detach().numpy()
                #print(pooled_feats.shape)
                # Ensure the feature dimension is as expected and add to the list
                if len(pooled_feats.shape) == 1:  # When batch_size is 1
                    all_feats.append(pooled_feats)
                elif len(pooled_feats.shape) == 2:  # General case
                    all_feats.extend(pooled_feats)
                else:
                    print(f"Unexpected shape of pooled features: {pooled_feats.shape}")
    
    return np.array(all_feats)


def get_all_image_feats_flava(dataloader, fe, model):
    """
    Gets image features from a dataloader and applies mean pooling to each set of image features.
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - fe: the feature extractor
    - model: the model used to extract image features
    Outputs:
    - a numpy array of shape [num_images, embed_dim] after mean pooling
    """
    all_feats = []
    with torch.no_grad():
        for d in dataloader:
            images_rgb = [image.convert("RGB") for image in d["images"]]
            image_input = fe(images_rgb, return_tensors="pt")
            feats = model.get_image_features(**image_input).detach().numpy()
            mean_feats = np.mean(feats, axis=1)  # Mean pooling across the patches
            all_feats.append(mean_feats)
    return np.concatenate(all_feats, axis=0)



def get_all_image_feats_cvcl(dataloader, preprocess, model):
    """
    Gets image features from a dataloader
    -----
    Inputs:
    - dataloader: a dataloader constructed from a DevBenchDataset
    - processor: the appropriate input processor for the model
    - model: the model used to extract image features
    - device: torch device (cuda or cpu)
    Outputs:
    - a numpy array of shape [num_images, embed_dim]
    """
    all_feats = []
    model.eval()
    with torch.no_grad():
        for d in dataloader:
            # Process the images with the processor
            images = [preprocess(img.convert("RGB")) for img in d["images"]]
            images = torch.stack(images).to(model.device)
            image_features = model.encode_image(images)
            print(image_features.shape)
            #processed_inputs = processor(images=d["images"], return_tensors="pt")
            #pixel_values = processed_inputs["pixel_values"]
            
            # Get image features using model's encode_image method
            #image_features = model.encode_image(pixel_values).detach().numpy()
            
            # Append features to the list
            all_feats.append(image_features)
    
    return np.concatenate(all_feats, axis=0)