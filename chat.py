from flask import Flask, request, render_template
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# Function to generate text with additional parameters for better coherence
def generate_text(prompt, max_length=50, temperature=0.7, top_k=50, top_p=0.95):
    inputs = tokenizer.encode(prompt, return_tensors='tf')
    outputs = model.generate(
        inputs, 
        max_length=min(max_length + len(inputs[0]), 100),  # Ensuring the total length doesn't exceed 100 tokens
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p, 
        num_return_sequences=1,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_text = None
    if request.method == 'POST':
        prompt = request.form['prompt']
        generated_text = generate_text(prompt, max_length=50)
    return render_template('index.html', generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)