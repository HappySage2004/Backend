from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def calculate_similarity(text1, text2):
    # Tokenize the text paragraphs
    inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True)

    # Generate embeddings
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # Calculate cosine similarity between embeddings
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)
    similarity_score = torch.cosine_similarity(embeddings1, embeddings2, dim=1)

    return similarity_score.item()

@app.route('/api/similarity', methods=['POST'])
def similarity_endpoint():
    data = request.get_json()
    text1 = data.get('text1')
    text2 = data.get('text2')

    if not text1 or not text2:
        return jsonify({'error': 'Both text1 and text2 are required.'}), 400

    similarity_score = calculate_similarity(text1, text2)
    return jsonify({'similarity_score': similarity_score})

if __name__ == '__main__':
    app.run(debug=True)