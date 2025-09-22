import os
import cv2
import numpy as np

# --- Configuration ---
# The size of each chip. This is used to calculate the bounding box.
CHIP_SIZE = 224
# The color and thickness of the bounding box (in BGR format).
# BGR (0, 0, 255) is red.
BOX_COLOR = (0, 0, 255)
BOX_THICKNESS = 2
# A threshold to determine if two bounding boxes are "close enough" to be merged.
# This value may need to be adjusted based on the specific dataset and chip size.
MERGE_THRESHOLD = 50 
# The string to use for the 'target_object' field in the label file.
TARGET_OBJECT = "Pond-2(Filled)"
# The filename for the single output label file.
OUTPUT_LABEL_FILENAME = "all_merged_labels.txt"


class DisjointSetUnion:
    """
    A simple implementation of the Disjoint Set Union (DSU) data structure.
    Used to group interconnected bounding boxes.
    """
    def __init__(self, n_elements):
        self.parent = list(range(n_elements))

    def find(self, i):
        """Finds the representative (root) of the set containing element i."""
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i]) # Path compression for efficiency
        return self.parent[i]

    def union(self, i, j):
        """Unites the sets containing elements i and j."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def boxes_are_close(box_a, box_b, threshold):
    """
    Checks if two bounding boxes are close enough (overlap or within threshold)
    to be considered part of the same cluster.
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # Expand box A by the threshold to check for proximity
    x1_a_exp = x1_a - threshold
    y1_a_exp = y1_a - threshold
    x2_a_exp = x2_a + threshold
    y2_a_exp = y2_a + threshold

    # A simple intersection check between the expanded box A and box B
    intersects = (x1_a_exp < x2_b and x2_a_exp > x1_b and
                  y1_a_exp < y2_b and y2_a_exp > y1_b)
    
    return intersects

def merge_final_overlaps(boxes: list):
    """
    Performs a final, iterative merge on any remaining overlapping boxes.
    This is a safety check to catch any overlaps missed by the initial DSU pass.
    
    Args:
        boxes (list): A list of bounding boxes (x1, y1, x2, y2).
        
    Returns:
        list: A new list of non-overlapping merged boxes.
    """
    if not boxes:
        return []

    merged = False
    while not merged:
        merged = True
        i = 0
        while i < len(boxes):
            j = i + 1
            while j < len(boxes):
                if boxes_are_close(boxes[i], boxes[j], threshold=0):
                    # Overlap found, merge the two boxes
                    merged_box = (
                        min(boxes[i][0], boxes[j][0]),
                        min(boxes[i][1], boxes[j][1]),
                        max(boxes[i][2], boxes[j][2]),
                        max(boxes[i][3], boxes[j][3])
                    )
                    # Replace one box with the merged box and remove the other
                    boxes[i] = merged_box
                    del boxes[j]
                    merged = False  # A merge occurred, so we need to restart the loop
                else:
                    j += 1
            i += 1
    return boxes

def merge_overlapping_boxes(boxes: list):
    """
    Merges a list of overlapping bounding boxes into a smaller set of combined boxes
    using the Disjoint Set Union (DSU) algorithm to find connected components,
    followed by a final overlap check.
    
    Args:
        boxes (list): A list of bounding boxes, where each box is a tuple (x1, y1, x2, y2).
        
    Returns:
        list: A new list of robustly merged bounding boxes.
    """
    if not boxes:
        return []

    num_boxes = len(boxes)
    dsu = DisjointSetUnion(num_boxes)

    # Step 1: Union sets for all overlapping/proximate boxes
    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            if boxes_are_close(boxes[i], boxes[j], MERGE_THRESHOLD):
                dsu.union(i, j)

    # Step 2: Group boxes by their representative (root of their set)
    clusters = {}
    for i in range(num_boxes):
        root = dsu.find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(boxes[i])

    # Step 3: Compute the merged bounding box for each cluster
    dsu_merged_boxes = []
    for root in clusters:
        cluster_boxes = clusters[root]
        
        # Initialize with the first box in the cluster
        min_x = cluster_boxes[0][0]
        min_y = cluster_boxes[0][1]
        max_x = cluster_boxes[0][2]
        max_y = cluster_boxes[0][3]

        # Expand to include all boxes in the cluster
        for box in cluster_boxes:
            min_x = min(min_x, box[0])
            min_y = min(min_y, box[1])
            max_x = max(max_x, box[2])
            max_y = max(max_y, box[3])
        
        dsu_merged_boxes.append((min_x, min_y, max_x, max_y))

    # Step 4: Perform a final check to merge any remaining overlaps
    final_merged_boxes = merge_final_overlaps(dsu_merged_boxes)

    return final_merged_boxes


def visualize_chip_bboxes(chip_folder: str, parent_images_folder: str):
    """
    Parses image filenames from a folder, groups chips by their parent image,
    merges overlapping bounding boxes using DSU, draws them on each parent image, and
    creates a single label file.

    Args:
        chip_folder (str): The path to the folder containing the chip images.
        parent_images_folder (str): The path to the folder containing the parent images.
    """
    
    # Check if the input folders exist.
    if not os.path.isdir(chip_folder):
        print(f"Error: The chip folder '{chip_folder}' does not exist.")
        return
    if not os.path.isdir(parent_images_folder):
        print(f"Error: The parent images folder '{parent_images_folder}' does not exist.")
        return

    # Create the output folder to save the visualized images.
    output_folder = r"D:\visual_retrieval\visualized_results_merged"
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to store parent image data.
    parent_images_data = {}

    print("Step 1: Parsing filenames and grouping chips by parent image...")
    # Iterate through all files in the chip folder.
    for filename in os.listdir(chip_folder):
        if filename.endswith(".jpg"):
            try:
                file_without_ext = os.path.splitext(filename)[0]
                
                if '_chip_' in file_without_ext:
                    basename_part, rest_of_filename = file_without_ext.split('_chip_', 1)
                    coord_parts = rest_of_filename.rsplit('_', 2)
                    
                    if len(coord_parts) == 3:
                        x1 = int(coord_parts[1])
                        y1 = int(coord_parts[2])
                        x2 = x1 + CHIP_SIZE
                        y2 = y1 + CHIP_SIZE
                        
                        if basename_part not in parent_images_data:
                            parent_images_data[basename_part] = []
                        parent_images_data[basename_part].append((x1, y1, x2, y2))
                    else:
                        print(f"Warning: Skipping file due to invalid suffix format: {filename}")
                else:
                    print(f"Warning: Skipping file due to invalid filename format (missing '_chip_'): {filename}")
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse coordinates from file '{filename}'. Error: {e}")

    print("Step 2: Drawing merged bounding boxes on parent images and saving...")
    # Prepare to write all labels to a single file.
    output_label_path = os.path.join(output_folder, OUTPUT_LABEL_FILENAME)
    
    with open(output_label_path, 'w') as f:
        # Iterate through the grouped data to process each parent image.
        for parent_basename, original_boxes in parent_images_data.items():
            parent_image_path = os.path.join(parent_images_folder, f"{parent_basename}.jpg")
            
            if not os.path.exists(parent_image_path):
                print(f"Warning: Parent image not found for '{parent_basename}' in '{parent_images_folder}'. Skipping.")
                continue
                
            parent_img = cv2.imread(parent_image_path)
            
            if parent_img is None:
                print(f"Error: Could not load image from '{parent_image_path}'. Skipping.")
                continue
            
            merged_boxes = merge_overlapping_boxes(original_boxes)

            # Draw a bounding box for each merged cluster on the parent image.
            for x1, y1, x2, y2 in merged_boxes:
                cv2.rectangle(parent_img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
                
                # Write the label data for the merged box to the single output file.
                line = f"{x1},{y1},{x2},{y2},{TARGET_OBJECT},{parent_basename}.jpg,0.0\n"
                f.write(line)
            
            # Save the modified image to the output folder.
            output_filename = f"{parent_basename}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, parent_img)
            print(f"Saved image: {output_path}")

    print("\nProcess completed successfully!")
    print(f"Visualized images have been saved to the '{output_folder}' folder.")
    print(f"All bounding box labels have been consolidated into a single file: '{output_label_path}'")


# Main execution block to run the script.
if __name__ == "__main__":
    # Specify the paths to your folders here.
    # Replace these with the actual paths on your system.
    chip_images_folder = r"D:\visual_retrieval\search_results"
    parent_images_folder = r"D:\visual_retrieval\folder_images"
    

    
    visualize_chip_bboxes(chip_images_folder, parent_images_folder)