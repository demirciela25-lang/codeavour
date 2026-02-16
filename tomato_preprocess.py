import os
import random
import shutil

def create_reduced_dataset(src_root, dst_root, max_images=2000):
    
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    
    for class_name in os.listdir(src_root):
        src_class_path = os.path.join(src_root, class_name)
        dst_class_path = os.path.join(dst_root, class_name)
        
        if not os.path.isdir(src_class_path):
            continue
        
        os.makedirs(dst_class_path, exist_ok=True)
        
        images = os.listdir(src_class_path)
        
        if len(images) > max_images:
            selected_images = random.sample(images, max_images)
        else:
            selected_images = images
        
        for img in selected_images:
            shutil.copy(
                os.path.join(src_class_path, img),
                os.path.join(dst_class_path, img)
            )
        
        print(f"{class_name}: {len(selected_images)} images copied")

# 🔹 CHANGE THESE PATHS
create_reduced_dataset("codeavour/domates/tomato_dataset/tomato_dataset_in/train", "codeavour/domates/tomato_dataset/tomato_dataset_in/train_small", 2000)
create_reduced_dataset("codeavour/domates/tomato_dataset/tomato_dataset_in/valid", "codeavour/domates/tomato_dataset/tomato_dataset_in/valid_small", 2000)
create_reduced_dataset("codeavour/domates/tomato_dataset/tomato_dataset_in/test", "codeavour/domates/tomato_dataset/tomato_dataset_in/test_small", 2000)

print("Reduced dataset created.")
