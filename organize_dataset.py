import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

# Read metadata
df = pd.read_csv("C:/Projects/Langchain_Model/Skin_Disease_Detection/dataset/HAM10000_metadata.csv")

# Split data
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['dx'],
    random_state=42
)

# Function to organize images
def organize_images(dataframe, split_name):

    for _, row in dataframe.iterrows():

        label = row['dx']
        image_name = row['image_id'] + ".jpg"

        source = os.path.abspath(
            os.path.join("C:/Projects/Langchain_Model/Skin_Disease_Detection/all_images", image_name)
        )

        destination_folder = os.path.join(
            "C:/Projects/Langchain_Model/Skin_Disease_Detection/datasett",
            split_name,
            label
        )

        os.makedirs(destination_folder, exist_ok=True)

        destination = os.path.join(
            destination_folder,
            image_name
        )

        shutil.copy(source, destination)

# Create train dataset
organize_images(train_df, "train")

# Create validation dataset
organize_images(val_df, "validation")

print("Dataset organized successfully!")