import os
import cv2
import numpy as np
import torch
import open_clip
from PIL import Image
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
# Match the configuration from your main script
BASE_DIR = Path('./')
FOLDER_IMAGES = Path(r"D:\visual_retrieval\folder_images")
FOLDER_VISUALIZED_RESULTS = Path(r"D:\visual_retrieval\visualized_results_merged")
QUERY_CHIP_PATH = Path(r"D:\visual_retrieval\query_chip\query_chip.jpg")
TARGET_OBJECT = "Pond-2(Filled)" 
# The name of the single label file from your previous script
INPUT_LABELS_FILE = FOLDER_VISUALIZED_RESULTS / "all_merged_labels.txt"
# The name of the output file for similarity scores
OUTPUT_SCORES_FILE = FOLDER_VISUALIZED_RESULTS / "merged_box_similarities.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_preprocess_image(image_path, preprocess_fn):
    """Loads an image, converts to RGB, and applies the OpenCLIP preprocess function."""
    try:
        img_pil = Image.open(image_path).convert('RGB')
        return preprocess_fn(img_pil).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"Error loading and preprocessing image at {image_path}: {e}")
        return None

def calculate_similarity_and_save():
    """
    Calculates the similarity between the query chip and the final merged bounding boxes.
    """
    if not os.path.exists(QUERY_CHIP_PATH):
        print(f"Error: Query chip not found at '{QUERY_CHIP_PATH}'. Please run the main GUI script first to generate it.")
        return
        
    if not os.path.exists(INPUT_LABELS_FILE):
        print(f"Error: Consolidated labels file not found at '{INPUT_LABELS_FILE}'. Please run the main GUI script first with the single output file modification.")
        return

    # Load the OpenCLIP model and preprocessor
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='laion2b_s32b_b82k'
        )
        model.to(DEVICE)
        model.eval()
        print("OpenCLIP model loaded successfully.")
    except Exception as e:
        print(f"Failed to load OpenCLIP model: {e}")
        return

    # Generate embedding for the query chip
    query_img_tensor = load_and_preprocess_image(QUERY_CHIP_PATH, preprocess)
    if query_img_tensor is None:
        return
        
    with torch.no_grad():
        query_embedding = model.encode_image(query_img_tensor)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
    
    print("Query chip embedding generated.")

    # Process each merged box and calculate similarity
    print("Processing merged boxes and calculating similarity...")
    all_scores = []
    
    with open(INPUT_LABELS_FILE, 'r') as f_in:
        for line in f_in:
            try:
                # Format: x_min,y_min,x_max,y_max,target_object,parent_image_name,similarity_score
                parts = line.strip().split(',')
                if len(parts) < 6:
                    print(f"Warning: Skipping malformed line in labels file: {line.strip()}")
                    continue
                
                x1, y1, x2, y2 = map(int, parts[0:4])
                parent_image_name = parts[5]
                
                parent_image_path = FOLDER_IMAGES / parent_image_name
                if not os.path.exists(parent_image_path):
                    print(f"Warning: Parent image not found for '{parent_image_name}'. Skipping.")
                    continue
                
                # Load the parent image and crop the merged box region
                parent_img = Image.open(parent_image_path).convert('RGB')
                merged_box_crop = parent_img.crop((x1, y1, x2, y2))
                
                # Resize the cropped image to 224x224 for the model
                merged_box_crop_resized = merged_box_crop.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Generate embedding for the merged box crop
                merged_box_tensor = preprocess(merged_box_crop_resized).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    merged_box_embedding = model.encode_image(merged_box_tensor)
                    merged_box_embedding /= merged_box_embedding.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.cpu().numpy(), 
                    merged_box_embedding.cpu().numpy()
                )[0][0]

                # Convert to a score (e.g., 0-100 scale)
                # The score needs to be based on the model's output. Cosine similarity is a good metric.
                # A value of 1.0 means identical, 0.0 means no similarity. Let's scale it to 0-100.
                similarity_score = (similarity + 1) / 2 
                
                # Store the result
                all_scores.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'parent_image_name': parent_image_name,
                    'similarity_score': similarity_score
                })
                
            except Exception as e:
                print(f"Error processing a line: {e}. Line was: {line.strip()}")
    
    # Save the results to the output file
    with open(OUTPUT_SCORES_FILE, 'w') as f_out:
        for score_data in all_scores:
            line = (
                f"{score_data['x1']},{score_data['y1']},{score_data['x2']},{score_data['y2']},"
                f"{TARGET_OBJECT},{score_data['parent_image_name']},"
                f"{score_data['similarity_score']:.2f}\n"
            )
            f_out.write(line)

    print(f"\nSimilarity calculation complete. Results saved to '{OUTPUT_SCORES_FILE}'")
    print(f"Total merged boxes processed: {len(all_scores)}")

if __name__ == '__main__':
    calculate_similarity_and_save()