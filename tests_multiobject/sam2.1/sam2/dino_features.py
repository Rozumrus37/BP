import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoImageProcessor, AutoModel
import faiss
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#Load CLIP model and processor
# processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

#Load DINOv2 model and processor
processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

#Retrieve all filenames
base_path = "/datagrid/personal/rozumrus/BP_dg/sam2.1_output"

image_paths = [
    "00000002_true.jpg", "00000002_small.jpg",  "00000002_full.jpg"
]

images = [os.path.join(base_path, img) for img in image_paths]


#Define a function that normalizes embeddings and add them to the index
def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

# def extract_features_clip(image):
#     with torch.no_grad():
#         inputs = processor_clip(images=image, return_tensors="pt").to(device)
#         image_features = model_clip.get_image_features(**inputs)
#         return image_features

def extract_features_dino(image):
    with torch.no_grad():
        inputs = processor_dino(images=image, return_tensors="pt").to(device)
        outputs = model_dino(**inputs)
        image_features = outputs.last_hidden_state
        return image_features.mean(dim=1)

#Create 2 indexes.
# index_clip = faiss.IndexFlatL2(512)
index_dino = faiss.IndexFlatL2(768)

#Iterate over the dataset to extract features X2 and store features in indexes
for image_path in images:
    print(image_path)
    img = Image.open(image_path).convert('RGB')
    # clip_features = extract_features_clip(img)
    # add_vector_to_index(clip_features,index_clip)
    dino_features = extract_features_dino(img)
    add_vector_to_index(dino_features,index_dino)

# #store the indexes locally
# faiss.write_index(index_clip,"clip.index")
faiss.write_index(index_dino,"dino.index")


#Input image
source=os.path.join(base_path, "00000001.jpg")

image = Image.open(source)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# #Load model and processor DINOv2 and CLIP
# processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

# #Extract features for CLIP
# with torch.no_grad():
#     inputs_clip = processor_clip(images=image, return_tensors="pt").to(device)
#     image_features_clip = model_clip.get_image_features(**inputs_clip)

#Extract features for DINOv2
with torch.no_grad():
    inputs_dino = processor_dino(images=image, return_tensors="pt").to(device)
    outputs_dino = model_dino(**inputs_dino)
    image_features_dino = outputs_dino.last_hidden_state
    image_features_dino = image_features_dino.mean(dim=1)

def normalizeL2(embeddings):
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    return vector

image_features_dino = normalizeL2(image_features_dino)
# image_features_clip = normalizeL2(image_features_clip)

#Search the top 5 images
# index_clip = faiss.read_index("clip.index")
index_dino = faiss.read_index("dino.index")

#Get distance and indexes of images associated
d_dino,i_dino = index_dino.search(image_features_dino,3)

print(i_dino, d_dino)
# d_clip,i_clip = index_clip.search(image_features_clip,5)

# import torch
# from torchvision import transforms
# from PIL import Image
# from transformers import AutoFeatureExtractor, AutoModel
# from torch.nn.functional import cosine_similarity

# # Load model and feature extractor
# model_name = "facebook/dino-vitb16"
# feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
# model.eval()

# # Preprocess image
# def preprocess_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     inputs = feature_extractor(images=image, return_tensors="pt")
#     return inputs

# # Extract features
# def extract_features(inputs):
#     with torch.no_grad():
#         outputs = model(**inputs)
#         return outputs.pooler_output

# # Compare two images
# image1_path = "image1.jpg"
# image2_path = "image2.jpg"

# inputs1 = preprocess_image(image1_path)
# inputs2 = preprocess_image(image2_path)

# features1 = extract_features(inputs1)
# features2 = extract_features(inputs2)

# # Calculate cosine similarity
# similarity = cosine_similarity(features1, features2)
# print("Cosine Similarity:", similarity.item())
