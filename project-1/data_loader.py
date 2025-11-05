from datasets import load_dataset

def prepare_datasets():
    dataset = load_dataset('spider')
    
    train_data = dataset['train']
    val_data = dataset['validation']
    
    def format_example(example):
        question = example['question']
        query = example['query']
        schema_text = ""
        
        if 'table_names' in example:
            tables = example['table_names']
            columns = example['column_names']
            for idx, table in enumerate(tables):
                cols = [c[1] for c in columns if c[0] == idx]
                schema_text += f"Table {table}: " + ", ".join(cols) + ". "
        
        input_text = f"Question: {question}\nSchema: {schema_text}"
        return {"input": input_text, "target": query}
    
    train_dataset = train_data.map(format_example)
    val_dataset = val_data.map(format_example)
    
    train_dataset = train_dataset.select_columns(['input', 'target'])
    val_dataset = val_dataset.select_columns(['input', 'target'])
    
    return train_dataset, val_dataset