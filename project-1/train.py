from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from config import (
    BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, NUM_EPOCHS, LEARNING_RATE,
    WARMUP_STEPS, OUTPUT_DIR, SAVE_DIR
)
from data_loader import prepare_datasets
from model_setup import load_model_and_tokenizer, apply_lora, prepare_for_training
from tokenization import tokenize_datasets

def main():
    print("Loading datasets...")
    train_dataset, validation_dataset = prepare_datasets()
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Applying LoRA...")
    model = apply_lora(model)
    
    print("Tokenizing datasets...")
    tokenized_train, tokenized_val = tokenize_datasets(
        train_dataset, validation_dataset, tokenizer
    )
    
    print("Preparing model for training...")
    model = prepare_for_training(model)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_steps=WARMUP_STEPS,
        weight_decay=0.01,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        gradient_checkpointing=False
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {SAVE_DIR}...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Done!")

if __name__ == "__main__":
    main()