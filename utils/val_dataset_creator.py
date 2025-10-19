import shutil
import random
import os
def create_validation_dataset():
    base_dir = 'data/chest_xray/train'
    val_dir  = 'data/chest_xray/val'
    val_split = 0.2
    os.makedirs(val_dir,exist_ok=True)
    try:
        for cls in ['PNEUMONIA','NORMAL']:
            train_cls_dir = os.path.join(base_dir,cls)
            val_cls_dir = os.path.join(val_dir,cls)
            os.makedirs(val_cls_dir,exist_ok=True)

            images = os.listdir(train_cls_dir)
            random.shuffle(images)
            val_count = int(len(images)*val_split)
            print(val_count)
            val_images=images[:val_count]

            for img in val_images:
                shutil.move(os.path.join(train_cls_dir,img), os.path.join(val_cls_dir,img))
        print("Validation set created successfully!")
    except Exception as e:
        print(f"Error {e}")

create_validation_dataset()