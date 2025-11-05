# Mistral-7B SQL Query Generation

Fine-tune Mistral-7B Instruct with QLoRA to map natural language questions to SQL queries using the Spider dataset.

## Project Structure

```
project/
├── config.py           # Configuration parameters
├── data_loader.py      # Dataset loading and preprocessing
├── model_setup.py      # Model initialization and LoRA setup
├── tokenization.py     # Tokenization functions
├── train.py            # Main training script
└── README.md           # This file
```

## Installation

### Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU)

### Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers bitsandbytes peft datasets
```

## Usage

```bash
python train.py
```

Training output and model will be saved to `./mistral-sql-lora/`

## Dataset

Uses HuggingFace Spider dataset:
- 7,000 training examples
- 1,034 validation examples
- Multiple databases with complex schemas

Automatically downloaded on first run.

## Model Configuration

### Current Settings
- **Model**: Mistral-7B-Instruct-v0.2
- **Quantization**: 4-bit NF4
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **Learning Rate**: 2e-4
- **Batch Size**: 4
- **Gradient Accumulation**: 2
- **Epochs**: 1
- **Max Sequence Length**: 256


## Evaluation Metrics

### Current Baseline
Expected performance after training:
- **Exact Match (EM)**: 65-72%
- **Valid SQL**: 80-85%
- **Execution Accuracy**: 70-75%

## Optimization Strategies for Improved Accuracy

### 1. **Data Quality & Preprocessing**
- **Add schema descriptions**: Include column data types and constraints
- **Few-shot examples**: Add 3-5 similar example pairs in prompt
- **Question augmentation**: Paraphrase questions in dataset
- **Filter ambiguous queries**: Remove queries with multiple valid solutions

**Implementation**: Update `format_example()` in `data_loader.py`

```python
schema_text += f"Table {table}: "
schema_text += ", ".join([f"{c}({types[i]})" for i, c in enumerate(cols)])
```

### 2. **Model & Training Configuration**
- **Increase LoRA rank**: Change `LORA_R = 128` (from 64) - better capacity
- **Increase epochs**: Change `NUM_EPOCHS = 3` - more iterations
- **Reduce learning rate**: Change `LEARNING_RATE = 1e-4` - more stable
- **Increase batch size**: Change `BATCH_SIZE = 8` (if VRAM allows)
- **Longer sequences**: Change `MAX_LENGTH = 512` - capture complex queries

**Tradeoff**: Higher values = better accuracy but slower training

### 3. **Prompt Engineering**
- **Add task description**: "Generate SQL query for database question"
- **Include constraints**: "Use INNER JOIN. Avoid subqueries"
- **Add output format**: "Output only SQL, no explanation"

Update prompt template in `format_example()`:

```python
input_text = f"""[INST] Generate SQL query.
Question: {question}
Schema: {schema_text}
Rules: Only return SQL. Use INNER JOIN when needed. [/INST]"""
```

### 4. **Target Modules Expansion**
- **Current**: `target_modules=["q_proj", "v_proj"]`
- **Better**: Add more layers for finer-tuning

Update in `model_setup.py`:

```python
target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```
**Impact Trade OFF**: +3-5% accuracy but slower training

### 5. **Advanced Training Techniques**
- **Gradient checkpointing**: Enable to save memory
- **Mixed precision**: Use `bf16=True` (already enabled)
- **Warmup ratio**: Increase `WARMUP_STEPS = 500` for stability
- **Learning rate schedule**: Add cosine schedule

Update `training_args`:

```python
lr_scheduler_type="cosine",
warmup_ratio=0.1,
save_strategy="steps",
save_steps=500,
eval_strategy="steps",
eval_steps=500
```

### 6. **Post-Processing & Validation**
- **SQL syntax validation**: Check if generated SQL is valid
- **Schema constraint checking**: Ensure used tables/columns exist
- **Beam search decoding**: Use `num_beams=4` during inference
- **Temperature sampling**: Reduce `temperature=0.3` for consistency

### 7. **Dataset Specific**
- **Increase training data**: Use extended datasets (WikiSQL, SEDE)
- **Domain-specific fine-tuning**: Focus on specific database types
- **Hard negative mining**: Include tricky examples
- **Balanced sampling**: Ensure query complexity distribution

## Inference

After training, use model:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(model, "./mistral-sql-lora")

tokenizer = AutoTokenizer.from_pretrained("./mistral-sql-lora")

question = "What is the name of the student with highest GPA?"
schema = "Table students: id, name, gpa"
prompt = f"[INST] Question: {question}\nSchema: {schema} [/INST]"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=256)
sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(sql_query)
```

## Troubleshooting

**Out of Memory**: Reduce `BATCH_SIZE` or `MAX_LENGTH` in config.py

**Slow Training**: Use smaller model (Mistral-7B instead of 13B) or reduce `MAX_LENGTH`

**Poor Accuracy**: Increase `NUM_EPOCHS`, add more training data, or use larger LoRA rank

## References

- [Spider Dataset](https://github.com/tatp22/spider)
- [Mistral AI](https://mistral.ai/)

## License

MIT
