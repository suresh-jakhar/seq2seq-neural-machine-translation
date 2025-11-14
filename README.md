---
title: French Translation API
emoji: 🇫🇷
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# French Translation API

English to French translation API powered by a custom-trained PyTorch Transformer model.

## Features

- 🤖 Custom-trained Neural Machine Translation model
- 🌐 RESTful API with Flask
- 💻 Beautiful web interface
- 🚀 Production-ready with Gunicorn

## Live Demo

Visit the web interface at: `http://your-deployment-url/`

## API Endpoints

### Translate Text
```bash
POST /translate
Content-Type: application/json

{
  "text": "Hello, how are you?"
}
```

**Response:**
```json
{
  "original_text": "Hello, how are you?",
  "translated_text": "comment allez-vous ?",
  "success": true
}
```

### Health Check
```bash
GET /health
```

## Local Development

### Prerequisites
- Python 3.10
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/suresh-jakhar/french_translation.git
cd french_translation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser:
```
http://localhost:5000
```

## Deployment

### Deploy to Heroku

1. Install Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create a new Heroku app:
```bash
heroku create your-app-name
```

4. Push to Heroku:
```bash
git push heroku main
```

5. Open your app:
```bash
heroku open
```

### Deploy to Render/Railway

1. Connect your GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `gunicorn app:app`
4. Deploy!

## Model Details

- **Architecture:** Transformer (4 encoder + 4 decoder layers)
- **Embedding Dimension:** 384
- **Attention Heads:** 6
- **Feedforward Dimension:** 1024
- **Tokenizer:** BPE (Byte Pair Encoding)
- **Vocabulary Size:** 6000 tokens

## Files

- `app.py` - Flask API server
- `index.html` - Web interface
- `best_local_transformer.pt` - Trained model weights (69 MB)
- `bpe_enfr.model` - Tokenizer model
- `requirements.txt` - Python dependencies
- `Procfile` - Heroku deployment configuration
- `runtime.txt` - Python version specification

## Tech Stack

- **Backend:** Flask, PyTorch
- **Frontend:** HTML, CSS, JavaScript
- **Tokenizer:** SentencePiece
- **Deployment:** Gunicorn WSGI server

## API Usage Examples

### cURL
```bash
curl -X POST http://localhost:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Good morning"}'
```

### Python
```python
import requests

response = requests.post(
    'http://localhost:5000/translate',
    json={'text': 'Good morning'}
)
print(response.json())
```

### JavaScript
```javascript
fetch('http://localhost:5000/translate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Good morning'})
})
.then(res => res.json())
.then(data => console.log(data));
```

## License

Apache License 2.0

## Author

Suresh Jakhar

## Acknowledgments

- Built with PyTorch
- Trained on English-French parallel corpus
