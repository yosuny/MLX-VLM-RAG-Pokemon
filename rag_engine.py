import os
import chromadb
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
import numpy as np

class RAGEngine:
    def __init__(self, db_path="./chroma_db", collection_name="image_rag"):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Initialize SigLIP Model
        print("Loading SigLIP model...")
        self.model_name = "google/siglip-so400m-patch14-384"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        print("SigLIP model loaded.")

    def get_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        
        # Normalize
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return outputs[0].numpy().tolist()

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
        # Normalize
        outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return outputs[0].numpy().tolist()

    def index_images(self, image_paths, custom_metadatas=None):
        embeddings = []
        ids = []
        metadatas = []
        
        for idx, path in enumerate(image_paths):
            print(f"Indexing {path}...")
            emb = self.get_image_embedding(path)
            embeddings.append(emb)
            ids.append(os.path.basename(path)) # Use filename as ID
            
            # Use custom metadata if provided, otherwise default to path only
            if custom_metadatas and idx < len(custom_metadatas):
                meta = custom_metadatas[idx]
                # Ensure path is included
                if "path" not in meta:
                    meta["path"] = path
                metadatas.append(meta)
            else:
                metadatas.append({"path": path})
            
        self.collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        print(f"Indexed {len(image_paths)} images.")

    def search(self, query, top_k=2):
        # Query can be text or image path
        if os.path.exists(query):
            query_emb = self.get_image_embedding(query)
        else:
            query_emb = self.get_text_embedding(query)
            
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        
        return results

if __name__ == "__main__":
    # Test
    rag = RAGEngine()
