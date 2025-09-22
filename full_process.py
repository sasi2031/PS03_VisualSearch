import os
import cv2
import numpy as np
import torch
import faiss
import pickle
import open_clip
from PIL import Image, ImageTk
from tkinter import Tk, Label, Button, Canvas, filedialog, messagebox, Toplevel, Scale, HORIZONTAL, StringVar, LEFT, X, VERTICAL, Scrollbar, Frame, END
from tkinter.ttk import Progressbar
from pathlib import Path
import shutil

# --- Configuration ---
BASE_DIR = Path('./')
FOLDER_IMAGES = Path(r"D:\visual_retrieval\folder_images")
FOLDER_CHIPS = Path(r"D:\visual_retrieval\folder_chips")
FOLDER_EMBEDDINGS = Path(r"D:\visual_retrieval\folder_embedding")
FOLDER_SAVE_RESULTS = Path(r"D:\visual_retrieval\search_results")

FAISS_PATH = Path(r"D:\visual_retrieval\faiss_data")
FAISS_INDEX_PATH = FAISS_PATH / 'faiss_index.bin'
FAISS_METADATA_PATH = FAISS_PATH / 'faiss_metadata.pkl'

CHIP_SIZE = (224, 224)
OVERLAP_PERCENT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

# --- 1. Core Data Processing Functions ---

def generate_chips(image_path, chip_size=CHIP_SIZE, overlap_percent=OVERLAP_PERCENT):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return [], [], []
    h, w, _ = img.shape
    chip_h, chip_w = chip_size
    overlap_h = int(chip_h * overlap_percent)
    overlap_w = int(chip_w * overlap_percent)
    chips = []
    coords = []
    img_wh = []
    pad_h = chip_h - (h % (chip_h - overlap_h)) if h % (chip_h - overlap_h) != 0 else 0
    pad_w = chip_w - (w % (chip_w - overlap_w)) if w % (chip_w - overlap_w) != 0 else 0
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    padded_h, padded_w, _ = padded_img.shape
    for y in range(0, padded_h - chip_h + 1, chip_h - overlap_h):
        for x in range(0, padded_w - chip_w + 1, chip_w - overlap_w):
            chip = padded_img[y:y+chip_h, x:x+chip_w]
            chips.append(chip)
            coords.append([x,y])
            img_wh.append([w,h])
    return chips, coords, img_wh


# def generate_chips(image_path, chip_size=CHIP_SIZE, overlap_percent=OVERLAP_PERCENT):
#     try:
#         img_pil = Image.open(image_path).convert('RGB')
#         img = np.array(img_pil)
#     except Exception as e:
#         print(f"Error: Could not read image at {image_path}. Reason: {e}")
#         return []

#     h, w, _ = img.shape
#     chip_h, chip_w = chip_size
#     overlap_h = int(chip_h * overlap_percent)
#     overlap_w = int(chip_w * overlap_percent)

#     chips = []
#     pad_h = chip_h - (h % (chip_h - overlap_h)) if h % (chip_h - overlap_h) != 0 else 0
#     pad_w = chip_w - (w % (chip_w - overlap_w)) if w % (chip_w - overlap_w) != 0 else 0
    
#     padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
#     padded_h, padded_w, _ = padded_img.shape

#     for y in range(0, padded_h - chip_h + 1, chip_h - overlap_h):
#         for x in range(0, padded_w - chip_w + 1, chip_w - overlap_w):
#             chip = padded_img[y:y+chip_h, x:x+chip_w]
#             chips.append(chip)
            
#     return chips


def process_all_images_and_generate_embeddings(app_instance):
    if not FOLDER_IMAGES.exists() or not os.listdir(FOLDER_IMAGES):
        messagebox.showerror("Error", "Image folder is empty or not found. Please add images to the 'folder_images' directory.")
        return

    os.makedirs(FOLDER_CHIPS, exist_ok=True)
    os.makedirs(FOLDER_EMBEDDINGS, exist_ok=True)

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14',
            pretrained='laion2b_s32b_b82k'
        )
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load OpenCLIP model: {e}")
        return

    image_files = [f for f in os.listdir(FOLDER_IMAGES) if f.lower().endswith(SUPPORTED_IMAGE_TYPES)]
    total_images = len(image_files)
    if not total_images:
        messagebox.showinfo("Info", "No supported image files found in the folder.")
        return

    progress_bar = Toplevel()
    progress_bar.title("Processing Images")
    progress_label = Label(progress_bar, text="Generating embeddings...")
    progress_label.pack(pady=10)
    progress = Progressbar(progress_bar, orient=HORIZONTAL, length=300, mode='determinate')
    progress.pack(pady=10)

    for i, img_name in enumerate(image_files):
        progress['value'] = (i + 1) * 100 / total_images
        progress_bar.update_idletasks()

        image_path = FOLDER_IMAGES / img_name
        img_base_name = os.path.splitext(img_name)[0]
        chip_folder_path = FOLDER_CHIPS / img_base_name
        os.makedirs(chip_folder_path, exist_ok=True)
        
        chips, coords, img_wh = generate_chips(image_path)
        if not chips: continue
        
        embedding_folder_path = FOLDER_EMBEDDINGS / img_base_name
        os.makedirs(embedding_folder_path, exist_ok=True)
        
        for j, chip_cv2 in enumerate(chips):
            chip_filename = f"{img_base_name}_chip_{j}_{img_wh[j][0]}_{img_wh[j][1]}_{coords[j][0]}_{coords[j][1]}.jpg"
            chip_pil = Image.fromarray(cv2.cvtColor(chip_cv2, cv2.COLOR_BGR2RGB))
            chip_img_tensor = preprocess(chip_pil).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                embedding = model.encode_image(chip_img_tensor)
                embedding /= embedding.norm(dim=-1, keepdim=True)
            
            emb_filename = os.path.splitext(chip_filename)[0] + '.pt'
            torch.save(embedding.squeeze().cpu(), embedding_folder_path / emb_filename)
            
            # --- ADDED CODE: Saving chip images in the chip folder ---
            chip_save_path = chip_folder_path / chip_filename
            chip_pil.save(chip_save_path)
            # ------------------------------------
    
    progress_bar.destroy()
    messagebox.showinfo("Success", "All images processed and embeddings saved!")
    
    app_instance.build_faiss_index_ui()

def setup_faiss_index():
    all_embeddings = []
    metadata = []
    
    if not FOLDER_EMBEDDINGS.exists() or not os.listdir(FOLDER_EMBEDDINGS):
        messagebox.showerror("Error", "Embedding folder is empty. Please run 'Process Images & Create Embeddings' first.")
        return None, None

    embedding_files = []
    for parent_img_folder in os.listdir(FOLDER_EMBEDDINGS):
        parent_folder_path = FOLDER_EMBEDDINGS / parent_img_folder
        if parent_folder_path.is_dir():
            embedding_files.extend(parent_folder_path.glob('*.pt'))

    if not embedding_files:
        messagebox.showerror("Error", "No embedding files found. Index cannot be built.")
        return None, None

    for emb_path in embedding_files:
        try:
            embedding = torch.load(emb_path).numpy().astype('float32')
            all_embeddings.append(embedding)
            metadata.append({'parent_image': emb_path.parent.name, 'chip_name': emb_path.name})
        except Exception as e:
            print(f"Error loading {emb_path}: {e}")
            
    embeddings_np = np.vstack(all_embeddings)
    dimension = embeddings_np.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    FAISS_PATH.mkdir(exist_ok=True)
    
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
        
    messagebox.showinfo("Success", "FAISS index built and saved successfully!")
    return index, metadata

def prepare_user_image(user_image):
    """
    Prepares the user's cropped image based on a set of rules.
    Returns a list of prepared images for embedding.
    """
    
    img_width, img_height = user_image.size
    prepared_images = []

    # 1. Box dimensions < 224x224
    if img_width < 224 and img_height < 224:
        new_image = Image.new('RGB', (224, 224), (0, 0, 0))
        paste_x = (224 - img_width) // 2
        paste_y = (224 - img_height) // 2
        new_image.paste(user_image, (paste_x, paste_y))
        prepared_images.append(new_image)
        
    # 2. Box dimensions between 224 and 512 (inclusive)
    elif (224 <= img_width <= 512) or (224 <= img_height <= 512):
        if img_width > img_height:
            # Pad height to match width, then resize
            new_size = (img_width, img_width)
            new_image = Image.new('RGB', new_size, (0, 0, 0))
            new_image.paste(user_image, (0, (img_width - img_height) // 2))
        else:
            # Pad width to match height, then resize
            new_size = (img_height, img_height)
            new_image = Image.new('RGB', new_size, (0, 0, 0))
            new_image.paste(user_image, ((img_height - img_width) // 2, 0))
            
        final_image = new_image.resize((224, 224), Image.Resampling.LANCZOS)
        prepared_images.append(final_image)

    # 3. Box dimensions > 512x512
    elif img_width > 512 and img_height > 512:
        chip_size = 512
        overlap = int(chip_size * 0.5)
        
        for y in range(0, img_height, chip_size - overlap):
            if y + chip_size > img_height: y = img_height - chip_size
            for x in range(0, img_width, chip_size - overlap):
                if x + chip_size > img_width: x = img_width - chip_size
                chip = user_image.crop((x, y, x + chip_size, y + chip_size))
                prepared_images.append(chip.resize((224, 224), Image.Resampling.LANCZOS))
                if x + chip_size >= img_width: break
            if y + chip_size >= img_height: break

    # 4. Mixed dimensions (one side > 512, one side < 512)
    elif (img_width > 512 and img_height <= 512) or (img_width <= 512 and img_height > 512):
        chip_size = 512
        overlap = int(chip_size * 0.5)
        
        if img_width > 512:
            new_height = 512
            new_image = Image.new('RGB', (img_width, new_height), (0, 0, 0))
            new_image.paste(user_image, (0, (new_height - img_height) // 2))
            
            for x in range(0, img_width, chip_size - overlap):
                if x + chip_size > img_width: x = img_width - chip_size
                chip = new_image.crop((x, 0, x + chip_size, new_height))
                prepared_images.append(chip.resize((224, 224), Image.Resampling.LANCZOS))
                if x + chip_size >= img_width: break

        elif img_height > 512:
            new_width = 512
            new_image = Image.new('RGB', (new_width, img_height), (0, 0, 0))
            new_image.paste(user_image, ((new_width - img_width) // 2, 0))

            for y in range(0, img_height, chip_size - overlap):
                if y + chip_size > img_height: y = img_height - chip_size
                chip = new_image.crop((0, y, new_width, y + chip_size))
                prepared_images.append(chip.resize((224, 224), Image.Resampling.LANCZOS))
                if y + chip_size >= img_height: break
                
    return prepared_images

def find_similar_images(user_image, faiss_index, metadata, model, preprocess):
    prepared_images = prepare_user_image(user_image)

    if not prepared_images:
        return []
    
    all_search_results = []
    with torch.no_grad():
        for img in prepared_images:
            img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            embedding = model.encode_image(img_tensor)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            user_embedding = embedding.squeeze().cpu().numpy().astype('float32')
            
            top_k = 100
            distances, indices = faiss_index.search(np.expand_dims(user_embedding, axis=0), top_k)
            
            for i, dist in zip(indices[0], distances[0]):
                if 0 <= i < len(metadata):
                    parent_image_name = metadata[i]['parent_image']
                    chip_name = metadata[i]['chip_name']
                    
                    original_image_path = None
                    for ext in SUPPORTED_IMAGE_TYPES:
                        potential_path = FOLDER_IMAGES / (parent_image_name + ext)
                        if potential_path.exists():
                            original_image_path = str(potential_path)
                            break

                    if original_image_path is None:
                        original_image_path = "Path not found"
                    
                    similarity_score = 25 * (4 - dist)
                    
                    all_search_results.append({
                        'parent_image_name': parent_image_name,
                        'parent_image_path': original_image_path,
                        'chip_name': chip_name,
                        'similarity_score': similarity_score
                    })
    
    return all_search_results

# --- 2. Tkinter UI and App Logic ---
class ImageSearchApp:
    def __init__(self, master):
        self.master = master
        master.title("Image-Based Object Search")
        master.geometry("1200x800")
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.original_image = None
        self.tk_image = None
        self.cropped_image = None
        self.faiss_index = None
        self.metadata = None
        self.model = None
        self.preprocess = None
        
        self.x1, self.y1, self.x2, self.y2 = 0, 0, 0, 0
        self.rect_id = None
        self.start_x, self.start_y = None, None

        self.create_widgets()
        self.load_model_and_index()

    def on_closing(self):
        if FAISS_INDEX_PATH.exists():
            #os.remove(FAISS_INDEX_PATH)
            print(f"Deleted FAISS index file: {FAISS_INDEX_PATH}")
        if FAISS_METADATA_PATH.exists():
            #os.remove(FAISS_METADATA_PATH)
            print(f"Deleted FAISS metadata file: {FAISS_METADATA_PATH}")
        self.master.destroy()

    def create_widgets(self):
        top_frame = Frame(self.master)
        top_frame.pack(fill=X, pady=10)

        Button(top_frame, text="Process Images & Create Embeddings", command=lambda: process_all_images_and_generate_embeddings(self)).pack(side=LEFT, padx=5)
        Button(top_frame, text="Load Image for Search", command=self.load_image_for_search).pack(side=LEFT, padx=5)
        
        self.canvas = Canvas(self.master, bg="grey", relief="sunken", borderwidth=2)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        bottom_frame = Frame(self.master)
        bottom_frame.pack(fill=X, pady=10)
        
        Button(bottom_frame, text="Search for Similar Images", command=self.perform_search_ui).pack(side=LEFT, padx=5)

        self.results_frame = Frame(self.master)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        Label(self.results_frame, text="Search Results:", font=("Arial", 14, "bold")).pack(anchor='w')
        
        self.result_canvas_frame = Frame(self.results_frame)
        self.result_canvas_frame.pack(fill="both", expand=True)
        self.result_canvas = Canvas(self.result_canvas_frame)
        self.result_scrollbar = Scrollbar(self.result_canvas_frame, orient=VERTICAL, command=self.result_canvas.yview)
        self.scrollable_frame = Frame(self.result_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.result_canvas.configure(
                scrollregion=self.result_canvas.bbox("all")
            )
        )
        self.result_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.result_canvas.configure(yscrollcommand=self.result_scrollbar.set)
        self.result_canvas.pack(side=LEFT, fill="both", expand=True)
        self.result_scrollbar.pack(side=LEFT, fill="y")

    def load_model_and_index(self):
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-L-14',
                pretrained='laion2b_s32b_b82k'
            )
            self.model.to(DEVICE)
            self.model.eval()
            
            if FAISS_INDEX_PATH.exists() and FAISS_METADATA_PATH.exists():
                self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
                with open(FAISS_METADATA_PATH, 'rb') as f:
                    self.metadata = pickle.load(f)
                messagebox.showinfo("Ready", "FAISS index and model loaded. Ready for search.")
            else:
                messagebox.showwarning("Warning", "FAISS index not found. Please process images and create embeddings first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load resources: {e}")

    def build_faiss_index_ui(self):
        self.faiss_index, self.metadata = setup_faiss_index()

    def load_image_for_search(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*"+";*".join(SUPPORTED_IMAGE_TYPES))])
        if file_path:
            self.original_image = Image.open(file_path).convert("RGB")
            self.display_image(self.original_image)
            
    def display_image(self, image):
        self.canvas.delete("all")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        img_width, img_height = image.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        self.resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(self.resized_image)
        self.canvas.create_image(canvas_width/2, canvas_height/2, image=self.tk_image, anchor="center")
        
        self.image_display_x = (canvas_width - new_width) / 2
        self.image_display_y = (canvas_height - new_height) / 2

    def start_draw(self, event):
        if self.original_image:
            self.start_x = event.x
            self.start_y = event.y
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def draw_rect(self, event):
        if self.original_image and self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def end_draw(self, event):
        if self.original_image:
            end_x, end_y = event.x, event.y
            self.x1 = int((min(self.start_x, end_x) - self.image_display_x) / (self.resized_image.width / self.original_image.width))
            self.y1 = int((min(self.start_y, end_y) - self.image_display_y) / (self.resized_image.height / self.original_image.height))
            self.x2 = int((max(self.start_x, end_x) - self.image_display_x) / (self.resized_image.width / self.original_image.width))
            self.y2 = int((max(self.start_y, end_y) - self.image_display_y) / (self.resized_image.height / self.original_image.height))
            
            img_w, img_h = self.original_image.size
            self.x1 = max(0, self.x1)
            self.y1 = max(0, self.y1)
            self.x2 = min(img_w, self.x2)
            self.y2 = min(img_h, self.y2)

            self.cropped_image = self.original_image.crop((self.x1, self.y1, self.x2, self.y2))
            
            if self.x1 >= self.x2 or self.y1 >= self.y2:
                self.cropped_image = None
                self.canvas.delete(self.rect_id)
                self.rect_id = None
                messagebox.showwarning("Invalid Selection", "Please draw a valid box on the image.")

    def perform_search_ui(self):
        if not self.cropped_image:
            messagebox.showerror("Error", "Please load an image and draw a box first.")
            return
        
        if not self.faiss_index or not self.metadata or not self.model:
            messagebox.showerror("Error", "Required resources not loaded. Please ensure you have processed images and built the index.")
            return

        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        pil_image = self.cropped_image
        opencv_image = np.array(pil_image)  
        cv2.imwrite(r"D:\visual_retrieval\query_chip\query_chip.jpg", opencv_image)

        similar_results = find_similar_images(self.cropped_image, self.faiss_index, self.metadata, self.model, self.preprocess)

        if similar_results:
            grouped_results = {}
            for result in similar_results:
                parent_name = result['parent_image_name']
                if parent_name not in grouped_results:
                    grouped_results[parent_name] = []
                grouped_results[parent_name].append(result)

            sorted_parent_names = sorted(grouped_results.keys(), 
                                         key=lambda name: max(res['similarity_score'] for res in grouped_results[name]), 
                                         reverse=True)
            
            # Create the directory for search results if it doesn't exist
            os.makedirs(FOLDER_SAVE_RESULTS, exist_ok=True)
            
            print("Saving chips started...")
            
            top_k_results = 100 # Save only the top 100 results
            results_to_save = []
            
            # Collect the top k results across all parent images
            for parent_name in sorted_parent_names:
                results_to_save.extend(sorted(grouped_results[parent_name], key=lambda x: x['similarity_score'], reverse=True))
            
            # Sort the entire list and take the top k
            results_to_save = sorted(results_to_save, key=lambda x: x['similarity_score'], reverse=True)[:top_k_results]

            # Display and save the top k results
            for result in results_to_save:
                parent_name = result['parent_image_name']
                path = result['parent_image_path']
                similarity_score = f"{result['similarity_score']:.2f}"
                
                print("chip name: ", result['chip_name'])
                chip_name = result['chip_name'].replace(".pt", ".jpg")
                print("corrected chip name: ", chip_name)
                
                # Display the result
                text_label = Label(self.scrollable_frame, text=f"-> Chip: {chip_name} | Path: {path} | Score: {similarity_score}", font=("Arial", 10), justify=LEFT, anchor="w")
                text_label.pack(fill=X, padx=15, pady=2)
                
                
                
                # Get source and destination paths for saving
                source_chip_path = FOLDER_CHIPS / parent_name / chip_name
                
                destination_path = FOLDER_SAVE_RESULTS / f"{chip_name}"
                
                # Save the chip
                try:
                    shutil.copy(source_chip_path, destination_path)
                    print(f"Saved chip to {destination_path}")
                except FileNotFoundError:
                    print(f"Error: Source chip not found at {source_chip_path}. Skipping save.")
                except Exception as e:
                    print(f"Error saving file {source_chip_path}: {e}")

            print("Saving chips ended.")
            
            messagebox.showinfo("Success", f"Search complete. The found chip images have been saved to {FOLDER_SAVE_RESULTS}")
        else:
            Label(self.scrollable_frame, text="No similar images found.", font=("Arial", 12, "italic")).pack(padx=5, pady=20)
            
if __name__ == '__main__':
    root = Tk()
    app = ImageSearchApp(root)
    root.mainloop()
