from config import MAX_LENGTH

def tokenize_fn(batch, tokenizer):
    combined = [inp + tgt for inp, tgt in zip(batch["input"], batch["target"])]
    
    encodings = tokenizer(
        combined,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

def tokenize_datasets(train_dataset, validation_dataset, tokenizer):
    def tokenize_wrapper(batch):
        return tokenize_fn(batch, tokenizer)
    
    tokenized_train = train_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_val = validation_dataset.map(
        tokenize_wrapper,
        batched=True,
        remove_columns=validation_dataset.column_names
    )
    
    return tokenized_train, tokenized_val