import os
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer
from datasets import load_dataset

tokenizer_path = r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Tokenizer\data\tokenizer\tokenizer.json"
output_path = Path(r"C:\Users\Ashmit Gupta\Desktop\Coding\Pytorch\Transformer\GPT\Professional GPT\Preprocess Data (Pre Train)")
os.makedirs(output_path, exist_ok=True)

def process_explicit_shuffle():
    enc = Tokenizer.from_file(tokenizer_path)
    bos_id = enc.token_to_id("<|bos|>")
    
    print("Loading raw dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split='train')
    
   
    print("Performing explicit index shuffle...")
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices) 
    
    split_idx = int(len(indices) * 0.99)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    splits = [('train', train_indices), ('val', val_indices)]

    for split_name, split_indices in splits:
        filename = os.path.join(output_path, f'{split_name}.bin')
        print(f"\nWriting shuffled {split_name} to {filename}...")
        
     
        with open(filename, 'wb') as f:
            batch_size = 500 
            token_buffer = []
            
            for i in tqdm(range(0, len(split_indices), batch_size)):
               
                batch_range = split_indices[i : i + batch_size]
                
                batch_examples = dataset.select(batch_range)
                
                for example in batch_examples:
                    text_ids = enc.encode(example['text']).ids
                    text_ids.insert(0, bos_id)
                    token_buffer.extend(text_ids)
                    
                    if len(token_buffer) >= 100000:
                        np_array = np.array(token_buffer, dtype=np.uint16)
                        f.write(np_array.tobytes())
                        token_buffer = [] 
                
            if token_buffer:
                np_array = np.array(token_buffer, dtype=np.uint16)
                f.write(np_array.tobytes())

    print("\nProcessing Complete!")
    print(f"Files saved in: {output_path}")

if __name__ == '__main__':
    process_explicit_shuffle()