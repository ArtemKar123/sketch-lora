import os
import json
import glob
import argparse
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# Define constants
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision"
GRID_SIZE = 100  # Higher resolution grid with separate x/y tokens
CACHE_DIR = "./hf/models/"  # Default cache directory for pretrained models

system_prompt="""You are an expert artist specializing in drawing sketches that are visually appealing, expressive, and professional.
You will be provided with a blank grid. Your task is to specify where to place strokes on the grid to create a visually appealing sketch of the given textual concept.
The grid uses numbers (1 to {res}) along the bottom (x axis) and numbers (1 to {res}) along the left edge (y axis) to reference specific locations within the grid. Each cell is uniquely identified by a combination of the corresponding x axis numbers and y axis number (e.g., the bottom-left cell is 'x1y1', the cell to its right is 'x2y1').
You can draw on this grid by specifying where to draw strokes. You can draw multiple strokes to depict the whole object, where different strokes compose different parts of the object. 
To draw a stroke on the grid, you need to specify the following:
Starting Point: Specify the starting point by giving the grid location (e.g., 'x1y1' for column 1, row 1).
Ending Point: Specify the ending point in the same way (e.g., 'x{res}y{res}' for column {res}, row {res}).
Intermediate Points: Specify at least two intermediate points that the stroke should pass through. List these in the order the stroke should follow, using the same grid location format (e.g., 'x6y5', 'x13y10' for points at column 6 row 5 and column 13 row 10).
Parameter Values (t): For each point (including the start and end points), specify a t value between 0 and 1 that defines the position along the stroke's path. t=0 for the starting point. t=1 for the ending point.
Intermediate points should have t values between 0 and 1 (e.g., "0.3 for x6y5, 0.7 for x13y10").

If you want to draw a big and long stroke, split it into multiple small curves that connect to each other.
These instructions will define a smooth stroke that follows a Bezier curve from the starting point to the ending point, passing through the specified intermediate points.
To draw a visually appealing sketch of the given object or concept, break down complex drawings into manageable steps. Begin with the most important part of the object, then observe your progress and add additional elements as needed. Continuously refine your sketch by starting with a basic structure and gradually adding complexity. Think step-by-step.

Provide the sketch in the following format with the following fields:
<formatting>
<concept>The concept depicted in the sketch.</concept>
<strokes>This element holds a collection of individual stroke elements that define the sketch. 
Each stroke is uniquely identified by its own tag (e.g., <s1>, <s2>, etc.).
Within each stroke element, there are three key pieces of information: 
<points>A list of x-y coordinates defining the curve. These points define the path the stroke follows.</points>
<t_values>A series of numerical timing values that correspond to the points. These values define the progression of the stroke over time, ranging from 0 to 1, indicating the order or speed at which the stroke is drawn.</t_values>
<id>A short descriptive identifier for the stroke, explaining which part of the sketch it corresponds to.</id>
</strokes>
</formatting>
"""


class SketchDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=1024, cache_file=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.system_prompt = system_prompt.format(res=GRID_SIZE)
        
        # Extract special tokens
        self.bos = self.tokenizer.bos_token or "<|begin_of_text|>"
        self.s_hdr = "<|start_header_id|>"
        self.e_hdr = "<|end_header_id|>"
        self.eot = "<|eot_id|>"
        
        # Try to load from cache if cache_file is provided
        if cache_file and os.path.exists(cache_file):
            print(f"Loading dataset from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                self.examples = pickle.load(f)
            print(f"Loaded {len(self.examples)} examples from cache")
            return
        
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            
            # Extract class and output
            for sample in data:
                prompt = sample['prompt']
                sketch_output = sample['answer']
                
                # Format as instruction
                formatted_input = self._format_instruction(prompt, sketch_output)
                
                # Tokenize
                tokenized = self.tokenize(formatted_input)
                if tokenized:
                    self.examples.append(tokenized)
        
        # Save to cache if cache_file is provided
        if cache_file:
            print(f"Saving dataset to cache: {cache_file}")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.examples, f)
            print(f"Saved {len(self.examples)} examples to cache")
    
    def _format_instruction(self, prompt, output):
        """Format data as instruction for model with system prompt using LLaMA 3 token format"""
        formatted = (
            f"{self.bos}"
            f"{self.s_hdr}system{self.e_hdr}\n{self.system_prompt}{self.eot}"
            f"{self.s_hdr}user{self.e_hdr}\n{prompt}{self.eot}"
            f"{self.s_hdr}assistant{self.e_hdr}\n{output}{self.eot}"
        )
        return formatted
    
    def tokenize(self, text):
        """Tokenize text with appropriate padding and truncation"""
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Make sure we don't exceed max length
        if len(encodings.input_ids[0]) > self.max_length:
            return None
            
        # Create labels (same as input_ids for causal LM)
        encodings['labels'] = encodings.input_ids.clone()
        
        # Return as dict with tensors
        return {
            'input_ids': encodings.input_ids[0],
            'attention_mask': encodings.attention_mask[0],
            'labels': encodings.labels[0]
        }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def prepare_model_and_tokenizer(model_name, use_4bit=True, use_nested_quant=True):
    """
    Load and prepare the model and tokenizer with proper quantization.
    """
    print(f"Loading model and tokenizer from {model_name}")
    
    # Configure BitsAndBytes for 4-bit quantization
    quantization_config = None
    if use_4bit:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=use_nested_quant,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["lm_head"],  # Skip quantizing the output layer
                llm_int8_threshold=6.0,  # Threshold for outlier detection
                llm_int8_has_fp16_weight=True  # Use fp16 weights for better stability
            )
            print("Successfully configured 4-bit quantization")
        except Exception as e:
            print(f"Warning: Could not initialize 4-bit quantization: {e}")
            print("Falling back to 16-bit precision")
            use_4bit = False
    
    # Load model with quantization and efficient memory mapping
    try:
        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True
        )
        print("Successfully loaded model with quantization")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Load processor and get tokenizer
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    tokenizer = processor.tokenizer
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        modules_to_save=["lm_head"],  # Save the output layer
        init_lora_weights="gaussian"  # Initialize LoRA weights with gaussian distribution
    )
    
    # Prepare model for LoRA fine-tuning with quantization
    if use_4bit:
        try:
            model = prepare_model_for_kbit_training(model)
            print("Successfully prepared model for 4-bit training")
        except Exception as e:
            print(f"Warning: Could not prepare model for 4-bit training: {e}")
            print("Falling back to 16-bit training")
            use_4bit = False
    
    # Apply LoRA
    try:
        model = get_peft_model(model, lora_config)
        print("Successfully applied LoRA configuration")
        
        # Verify trainable parameters
        trainable_params = 0
        all_param = 0
        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable parameter: {name}")
        
        print(f"Trainable params: {trainable_params:,d} || All params: {all_param:,d} || Trainable%: {100 * trainable_params / all_param:.2f}%")
        
        # Ensure base model parameters are frozen
        for name, param in model.named_parameters():
            if "lora" not in name and "lm_head" not in name:
                param.requires_grad = False
        
        # Double check LoRA parameters are trainable
        for name, param in model.named_parameters():
            if "lora" in name:
                if not param.requires_grad:
                    print(f"Warning: LoRA parameter {name} is not trainable!")
                    param.requires_grad = True
        
    except Exception as e:
        print(f"Error applying LoRA: {e}")
        raise
    
    return model, tokenizer

def train(args):
    """Main training function"""
    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(
        args.model_name, 
        use_4bit=args.use_4bit, 
        use_nested_quant=args.use_nested_quant
    )
    
    # Create dataset with caching
    cache_file = os.path.join(args.output_dir, "dataset_cache.pkl")
    dataset = SketchDataset(args.data_dir, tokenizer, max_length=args.max_length, cache_file=cache_file)
    
    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        fp16=True,
        report_to="tensorboard",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",  # Use the more memory-efficient optimizer
        remove_unused_columns=False,
        label_names=["labels"],  # Explicitly set label names
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LoRA sketch generation model")
    parser.add_argument("--data_dir", type=str, default="./processed_data", help="Directory containing processed sketch data")
    parser.add_argument("--output_dir", type=str, default="./sketch_lora_output", help="Directory to save model outputs")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME, help="Base model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use 4-bit quantization")
    parser.add_argument("--use_nested_quant", action="store_true", default=True, help="Use nested quantization for 4-bit")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args) 