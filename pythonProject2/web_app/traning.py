from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# AI Hub 데이터셋 로드
def load_custom_dataset(train_path, val_path):
    try:
        train_dataset = load_dataset('json', data_files={'train': train_path})
        val_dataset = load_dataset('json', data_files={'validation': val_path})
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None, None
    return train_dataset, val_dataset

# 모델 및 토크나이저 설정
model_name = "facebook/bart-large-cnn"  # BART 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# 데이터 전처리 함수
def tokenize_function(examples):
    inputs = examples["passage"]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 데이터셋 로드 및 전처리
train_path = './Trainset.json'
val_path = './Validationdataset.json'
train_dataset, val_dataset = load_custom_dataset(train_path, val_path)

if train_dataset is None or val_dataset is None:
    print("Failed to load dataset. Exiting.")
    exit()

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# 데이터 콜레이터 설정
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 데이터 로더 설정
train_dataloader = DataLoader(tokenized_train_dataset["train"], shuffle=True, collate_fn=data_collator, batch_size=8)
eval_dataloader = DataLoader(tokenized_val_dataset["validation"], collate_fn=data_collator, batch_size=8)

# 훈련 인자 설정
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
)

# 트레이너 설정 및 훈련
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset["train"],
    eval_dataset=tokenized_val_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 예외 처리 및 디버깅을 위한 코드 추가
try:
    trainer.train()
except Exception as e:
    print(f"An error occurred during training: {str(e)}")

# 훈련 완료 후 평가
try:
    trainer.evaluate()
except Exception as e:
    print(f"An error occurred during evaluation: {str(e)}")


# 모델 및 토크나이저 저장
try:
    model.save_pretrained("./models/")
    tokenizer.save_pretrained("./models/", push_to_hub=False, tokenizer_name="my_custom_tokenizer")
    print("Model and tokenizer saved successfully.")
except Exception as e:
    print(f"An error occurred while saving model and tokenizer: {str(e)}")
