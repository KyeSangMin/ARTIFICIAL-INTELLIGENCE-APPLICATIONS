from flask import Flask, request, render_template, jsonify
import subprocess

app = Flask(__name__)

def summarize_text_with_script(text, method):
    text = text.encode('utf-8').decode('utf-8')  # Ensure the text is properly encoded
    result = subprocess.run(['python', 'summarize.py', method, text], capture_output=True, text=True, encoding='utf-8')
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        error_message = f"Error: Unable to summarize the text.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        return error_message

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    ollama_summary = summarize_text_with_script(text, 'ollama')
    transformers_summary = summarize_text_with_script(text, 'pipeline')
    custom_model_summary = summarize_text_with_script(text, 'custom_model')
    return jsonify({
        'ollama_summary': ollama_summary,
        'transformers_summary': transformers_summary,
        'custom_model_summary': custom_model_summary
    })

if __name__ == '__main__':
    app.run(debug=True)
