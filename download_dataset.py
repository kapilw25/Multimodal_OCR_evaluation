from datasets import load_dataset
import os

# Create dataset directory if it doesn't exist
os.makedirs("dataset", exist_ok=True)

print("Starting to download the Multi-modal-Self-instruct dataset...")
dataset = load_dataset("zwq2018/Multi-modal-Self-instruct")
print("Dataset downloaded successfully!")

print("Saving dataset to disk...")
dataset.save_to_disk("./dataset")
print("Dataset saved to ./dataset directory!")

# Print some information about the dataset
print("\nDataset Information:")
print(f"Available splits: {dataset.keys()}")
for split in dataset.keys():
    print(f"Split '{split}' contains {len(dataset[split])} examples")
    print(f"Features: {dataset[split].features}")
    print(f"First example: {dataset[split][0]}")
