import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import ollama

# Load the custom trained model and tokenize
model_name_or_path = "./models/"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

def summarize_with_custom_model(text):
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_with_pipeline(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=128, min_length=30, do_sample=False)[0]['summary_text']
    return summary

def summarize_with_ollama(text):
    # Replace with actual ollama summarization call
    response = ollama.chat(model='llama3:latest',messages=[{'role': 'user', 'content': f"Please summarize the following text: {text}"}])
    summary = response['message']['content']
    return summary

def main():
    method = sys.argv[1]
    text = sys.argv[2]

    if method == 'custom_model':
        print(summarize_with_custom_model(text))
    elif method == 'pipeline':
        print(summarize_with_pipeline(text))
    elif method == 'ollama':
        print(summarize_with_ollama(text))
    else:
        print("Invalid method specified.")

if __name__ == "__main__":
    main()
