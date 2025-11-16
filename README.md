# Seq2Seq Neural Machine Translation

English to French translation using custom Transformer model.

**Author:** Suresh Jakhar

**Live:** https://sureshjakhar-seq2seq-neural-machine-translation.hf.space

---

## API Reference

**Base URL:** `https://sureshjakhar-seq2seq-neural-machine-translation.hf.space`

### Authentication

No authentication required. This API is publicly accessible.

### Endpoints

#### POST /translate

Translate text from English to French using our custom-trained Transformer model.

**Request Body:**
```json
{
  "text": "string"
}
```

- `text` (string, required): English text to translate (max 48 tokens)

**Response:**
```json
{
  "original_text": "string",
  "success": true,
  "translated_text": "string"
}
```

**Example Request:**
```bash
curl -X POST https://sureshjakhar-seq2seq-neural-machine-translation.hf.space/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?"
  }'
```

**Example Response:**
```json
{
  "original_text": "Hello, how are you today?",
  "success": true,
  "translated_text": "bonjour, comment allez-vous aujourd'hui?"
}
```

#### GET /health

Check the health status of the translation service.

**Response:**
```json
{
  "device": "cpu",
  "model_loaded": true,
  "status": "healthy",
  "tokenizer_loaded": true
}
```

#### GET /api

Retrieve API metadata and available endpoints.

**Response:**
```json
{
  "endpoints": {
    "/health": "GET - Health check",
    "/translate": "POST - Translate text to French"
  },
  "message": "Seq2Seq Neural Machine Translation",
  "version": "1.0"
}
```

### Error Handling

**Error Response Format:**
```json
{
  "error": "string",
  "message": "Detailed error description"
}
```

**Common Error Codes:**
- **400 Bad Request** - No data provided or missing "text" field
- **500 Internal Server Error** - Translation failed during processing
- **503 Service Unavailable** - Translation model not loaded


## Technical Overview

### Architecture
- **Model:** Transformer (4 encoder layers, 4 decoder layers)
- **Hidden Dimensions:** 384 (d_model), 1024 (feedforward)
- **Attention Heads:** 6
- **Dropout:** 0.15
- **Parameters:** ~20M trainable parameters
- **Tokenization:** SentencePiece BPE (8,000 vocabulary)
- **Max Sequence Length:** 50 tokens

### Training
- **Dataset:** 200,000 parallel English-French sentence pairs
- **Optimizer:** AdamW with Noam warmup schedule
- **Techniques:** Gradient accumulation, mixed precision training, label smoothing
- **Hardware:** Mid-range laptop with 8GB RAM (CPU-only, no dedicated GPU)
- **Framework:** PyTorch 2.5.1

### Deployment
- **API:** Flask 3.0.0
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Hosting:** Hugging Face Spaces
- **Testing:** pytest with 92% code coverage

---

## Project Structure

```
seq2seq-neural-machine-translation/
├── app.py                       # Flask API server
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── runtime.txt                  # Python version specification
├── models/                      # Model artifacts (Git LFS)
│   ├── best_local_transformer.pt
│   └── bpe_enfr.model
├── static/                      # Web interface
│   └── index.html
├── tests/                       # Test suite
│   ├── test_app.py
│   └── test_model.py
├── .github/workflows/
│   └── ci-cd.yml               # Automated testing and deployment
└── training/                    # Training notebooks and data
    ├── english_to_french.ipynb
    └── eng-french.csv
```

---

## About This Project

### Training of Translation Model

Training a translation model from scratch on a mid-range laptop with 8GB RAM and no dedicated GPU. The dataset contained 200,000 parallel English-French sentences—modest by machine translation standards, but large enough that every design decision mattered for feasibility on CPU-only hardware.

The first challenge was representation. Word-level models create massive vocabularies and produce many unknown tokens for rare words. I used SentencePiece with the unigram language model to train a subword tokenizer on the combined English-French corpus. The resulting 8,000-token vocabulary provided efficient encoding while maintaining good generalization on the small dataset.

With the tokenizer in place, the model itself took shape not a giant pretrained system, but a compact Transformer the student could actually train. Four encoder layers, four decoder layers, a hidden dimension of 384, six attention heads, and feed forward networks scaled to 1,024 units. It resembled the original Transformer architecture in structure but was pared down deliberately to avoid memory issues. Positional encodings remained sinusoidal, and the embedding matrix was tied to the output projection a small but important act of parameter discipline.

Initial attempts crashed with out-of-memory errors; when they ran, gradients exploded or vanished. The solution was gradient accumulation: processing multiple micro-batches and summing their gradients before each optimizer step. This approximated the gradient behavior of larger batches while fitting in memory, though it doesn't replicate true batch-level dynamics.

AdamW provided decoupled weight decay, while the Noam warmup schedule stabilized early training. Automatic mixed precision reduced memory usage, though it required gradient scaling to prevent numerical instability. Label smoothing regularized the decoder. Two types of attention masks were critical: causal masks prevented the decoder from attending to future tokens, and padding masks excluded padded positions from attention computations.

The model began to translate, gradually improving with each epoch. What emerged was not a production grade system but a functioning demonstration of how a translation model can be built from scratch.

I produced several artifacts: a trained SentencePiece tokenizer, a compact Transformer checkpoint, and a translation engine capable of converting English sentences into French. The real achievement was understanding how tokenization shapes model behavior, how architecture must bend to hardware limits, how learning rate schedules prevent collapse, and how stability techniques such as gradient accumulation turn an otherwise impossible task into a successful experiment.

A practical lesson in modern NLP engineering: even with limited resources, a well designed model, a disciplined training strategy, and the willingness to iterate can produce a translation system that works.

### Deployment and MLOps

Training a model is one thing. Making it accessible, testable, and deployable is another entirely. The challenge: how to take a model file sitting on a local machine and turn it into something that runs reliably, can be tested automatically, and deploys without manual intervention. The first instinct was simple: write a Flask API, wrap the model in an endpoint, and call it done. A POST /translate route accepted English text, ran it through the Transformer, and returned French. Locally it worked, but sharing it became complicated.

GitHub warns about files over 50MB. The 72MB model file and tokenizer needed Git LFS for efficient version control. I configured .gitattributes to track *.pt and *.model files through LFS, which stores large files externally while keeping the repository lightweight.

Then came testing with pytest: health endpoint, valid inputs, empty strings, missing fields, JSON edge cases. Each test defined a contract. GitHub Actions ran tests automatically on every push. For Hugging Face Spaces, I configured the Space to use Docker mode (not auto-detected), which builds from the Dockerfile. The 176MB dataset couldn't live in the Spaces repo—best practice is uploading datasets to the Hugging Face datasets hub separately. I used .hfignore to exclude it from deployment and git filter-repo to clean the history.

The result was an automated pipeline. Code pushed to GitHub triggers automated tests. If tests pass, a Docker image builds and pushes to Docker Hub. Simultaneously, Hugging Face Spaces builds and deploys the live demo.

### What I Learned

This project taught me the gap between a local model and a production system. Limited RAM (no GPU) forced deliberate choices: fewer layers, gradient accumulation, and treating tokenization and learning rates as core architectural decisions. These constraints made efficiency mandatory. Deployment taught different lessons: APIs require error handling and clear contracts; Git LFS handles large files; git filter-repo cleans history; and continuous testing prevents regressions. Hugging Face Spaces needs explicit Docker configuration, not automatic detection. Each layer—training, testing, deployment—demanded its own rigor.

Automation, containerization, and deployment completed the education. CI/CD pipelines enforce consistency. Docker provides reproducibility. Deploying to Hugging Face and Docker Hub requires understanding authentication, file limits, and configurations. Debugging logs and refining workflows showed that system reliability depends on discipline and iteration. Building ML systems extends far beyond training models—it requires infrastructure, automation, and engineering to make systems stable, accessible, and maintainable.

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/suresh-jakhar/seq2seq-neural-machine-translation.git
cd seq2seq-neural-machine-translation

# Install dependencies 
pip install -r requirements.txt

# Run locally
python app.py

# Visit http://localhost:5000
```

### Docker Deployment

```bash
# Pull from Docker Hub
docker pull sureshjakhar/seq2seq-neural-machine-translation:latest

# Run container
docker run -p 5000:5000 sureshjakhar/seq2seq-neural-machine-translation:latest
```

---

## License

Apache-2.0 License

---

## Contact

**Suresh Jakhar**
- GitHub: [@suresh-jakhar](https://github.com/suresh-jakhar)
- Email: gettingsuresh@gmail.com
