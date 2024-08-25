import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import pipeline

# transcriber = pipeline(model="openai/whisper-large-v2", device=0)

# print(transcriber("./mlk.flac"))

# def data():
#     for i in range(1000):
#         yield f"My example {i}"


# pipe = pipeline(model="openai-community/gpt2", device=0)
# generated_characters = 0
# for out in pipe(data()):
#     generated_characters += len(out[0]["generated_text"])
#     print(out[0]["generated_text"])

from datasets import load_dataset
dataset = load_dataset("yelp_review_full") #原始数据集

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased") #加载分词器

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5) #yelp数据集中每个评论有5个类别，因此model的输出层要有5个神经元，并输出5个值

from transformers import TrainingArguments
training_args = TrainingArguments(output_dir="test_trainer") #default

import numpy as np
import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    tmp = metric.compute(predictions=predictions, references=labels)
    print(tmp)
    return tmp

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

trainer = Trainer(model = model, 
                  args = training_args, 
                  train_dataset = small_train_dataset,
                  eval_dataset = small_eval_dataset,
                  compute_metrics = compute_metrics,)
trainer.train()