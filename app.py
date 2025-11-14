import os
import torch
import torch.nn as nn
import sentencepiece as spm
import math
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Model configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
tokenizer = None
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
MAX_LEN = 50

# Define the Transformer model architecture
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.15, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=6, 
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        self.es = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.et = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.ps = PositionalEncoding(d_model, dropout)
        self.pt = PositionalEncoding(d_model, dropout)
        
        self.tr = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.et.weight  # Shared embedding weights
    
    @staticmethod
    def look_ahead_mask(T):
        return torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        
    def forward(self, src, tgt):
        src_pad_mask = (src == PAD_ID)
        tgt_pad_mask = (tgt == PAD_ID)
        tgt_mask = self.look_ahead_mask(tgt.size(1)).to(src.device)
        
        src_emb = self.ps(self.es(src))
        tgt_emb = self.pt(self.et(tgt))
        
        output = self.tr(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        
        return self.out(output)

def load_model():
    """Load the translation model and tokenizer"""
    global model, tokenizer
    try:
        # Load tokenizer
        tokenizer_path = 'bpe_enfr.model'
        if os.path.exists(tokenizer_path):
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(tokenizer_path)
            print(f"Tokenizer loaded successfully from {tokenizer_path}")
        else:
            print(f"Warning: Tokenizer file not found at {tokenizer_path}")
            return
        
        # Load model checkpoint
        model_path = 'best_local_transformer.pt'
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            return
            
        state = torch.load(model_path, map_location=device, weights_only=True)
        
        # Extract model parameters from checkpoint
        vocab_size = state["es.weight"].shape[0]
        d_model = state["es.weight"].shape[1]
        dim_feedforward = state["tr.encoder.layers.0.linear1.weight"].shape[0]
        
        # Initialize model with correct parameters
        model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=6,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=dim_feedforward,
            dropout=0.15
        )
        
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

def encode_sentence(text):
    """Encode text to tensor with proper padding"""
    ids = [BOS_ID] + tokenizer.encode(text.lower(), out_type=int)[:MAX_LEN-2] + [EOS_ID]
    ids = ids + [PAD_ID] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

def decode_tokens(tokens):
    """Decode token IDs to text"""
    ids = tokens.tolist()
    if EOS_ID in ids:
        ids = ids[:ids.index(EOS_ID)]
    return tokenizer.decode(ids)

def translate_text(text, max_length=MAX_LEN):
    """Translate English text to French"""
    if model is None or tokenizer is None:
        raise ValueError("Model or tokenizer not loaded")
    
    with torch.no_grad():
        src = encode_sentence(text)
        tgt = torch.tensor([[BOS_ID]], dtype=torch.long).to(device)
        
        for _ in range(max_length - 1):
            logits = model(src, tgt)
            next_token = logits[0, -1].argmax().item()
            tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
            if next_token == EOS_ID:
                break
        
        translation = decode_tokens(tgt[0, 1:])
        return translation

load_model()

@app.route('/')
def home():
    """Serve the frontend HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/api')
def api_info():
    """API information"""
    return jsonify({
        'message': 'French Translation API',
        'version': '1.0',
        'endpoints': {
            '/translate': 'POST - Translate text to French',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'device': str(device)
    }), 200

@app.route('/translate', methods=['POST'])
def translate():
    """Translate text to French"""
    try:
        # Check if model is loaded
        if model is None or tokenizer is None:
            return jsonify({
                'error': 'Translation model not loaded',
                'message': 'Please ensure the model and tokenizer files exist'
            }), 503
        
        # Get the input data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide JSON data with a "text" field'
            }), 400
        
        # Extract text from request
        text = data.get('text') or data.get('data')
        
        if not text:
            return jsonify({
                'error': 'No text provided',
                'message': 'Please provide text in the "text" or "data" field'
            }), 400
        
        # Perform translation
        translated_text = translate_text(text)
        
        return jsonify({
            'original_text': text,
            'translated_text': translated_text,
            'success': True
        }), 200
        
    except Exception as e:
        import traceback
        print(f"Translation error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': 'Translation failed',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
