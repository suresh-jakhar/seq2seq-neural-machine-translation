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
- **Hardware:** Mid-range laptop with limited VRAM
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

An attempt to understand what it actually takes to train a translation model from scratch on a mid range laptop with barely enough VRAM to run a modern game. The dataset was not enormous by machine translation standards roughly two lakh parallel English French sentences but it was large enough to make the task real, and small enough that every design decision mattered.

The first challenge was representation. Word level models were out of the question: they explode in vocabulary size and collapse on rare words. Instead, the i adopted a SentencePiece byte pair encoding tokenizer and trained it directly on the combined English French corpus. This joint subword model carved both languages into a shared space of 8,000 tokens, small enough to fit on limited hardware yet expressive enough to preserve linguistic structure.

With the tokenizer in place, the model itself took shape not a giant pretrained system, but a compact Transformer the student could actually train. Four encoder layers, four decoder layers, a hidden dimension of 384, six attention heads, and feed forward networks scaled to 1,024 units. It resembled the original Transformer architecture in structure but was pared down deliberately to avoid memory issues. Positional encodings remained sinusoidal, and the embedding matrix was tied to the output projection a small but important act of parameter discipline.

The first attempts crashed with out of memory errors; even when they ran, gradients exploded or drifted unpredictably. The solution was not to shrink the architecture, but to re think how updates were computed. Gradient accumulation became the central mechanism processing several micro batches and combining their gradients before each optimization step. This allowed the model to behave as if it were training with a large batch even though only a small batch fit in memory.

Optimization followed a similarly careful path. AdamW provided weight decoupled regularization, while the Noam warmup schedule stabilized the early phase of training where Transformers are most fragile. Automatic mixed precision helped the model fit in VRAM without sacrificing numerical behavior. Label smoothing softened the learning signal and prevented the decoder from becoming overconfident. Attention masks prevented the model from attending to future tokens or padded positions, maintaining the mathematical integrity of the sequence-to-sequence formulation.

The model began to translate, gradually improving with each epoch. What emerged was not a production grade system but a functioning demonstration of how a translation model can be built from scratch.

I produced several artifacts: a trained SentencePiece tokenizer, a compact Transformer checkpoint, and a translation engine capable of converting English sentences into French. The real achievement was understanding how tokenization shapes model behavior, how architecture must bend to hardware limits, how learning rate schedules prevent collapse, and how stability techniques such as gradient accumulation turn an otherwise impossible task into a successful experiment.

A practical lesson in modern NLP engineering: even with limited resources, a well designed model, a disciplined training strategy, and the willingness to iterate can produce a translation system that works.

### Deployment and MLOps

Training a model is one thing. Making it accessible, testable, and deployable is another entirely. The challenge: how to take a model file sitting on a local machine and turn it into something that runs reliably, can be tested automatically, and deploys without manual intervention. The first instinct was simple: write a Flask API, wrap the model in an endpoint, and call it done. A POST /translate route accepted English text, ran it through the Transformer, and returned French. Locally it worked, but sharing it became complicated.

GitHub was the starting point. But the model file was 72 megabytes, and the tokenizer another few megabytes. Git refused to accept them. The solution was Git LFS for versioning large files. Configuring it meant creating a .gitattributes file, marking *.pt and *.model as LFS tracked. It was tedious but necessary.

Then came testing with pytest: health endpoint, valid inputs, empty strings, missing fields, JSON edge cases. Each test was a contract: if this input goes in, this behavior must come out. Running them before every commit required automation. GitHub Actions provided continuous integration: every push triggers an automated workflow that installs dependencies, executes tests, and reports results. Once tests passed, deployment to Hugging Face Spaces was next. Push a repository, it detects a Dockerfile, builds a container, and serves the app. Files over 10 megabytes must be tracked by Git LFS. The 176-megabyte training dataset violated this rule. The fix required rewriting git history with git filter branch and using a .hfignore file. It was messy, but it worked.

The result was an automated pipeline. Code pushed to GitHub triggers automated tests. If tests pass, a Docker image builds and pushes to Docker Hub. Simultaneously, Hugging Face Spaces builds and deploys the live demo.

### What I Learned

This project taught me the real gap between a model that runs locally and a system that works reliably. Limited VRAM forced deliberate design: choosing fewer layers, using gradient accumulation, and treating tokenization, learning rate schedules, and precision as architectural decisions, not routine settings. Those constraints shaped how I approached optimization and made efficiency a requirement, not an afterthought. But once the model trained, the real lessons began APIs, error handling, logging, and clear contracts turned it from an experiment into something others could actually use. Git taught me that large files require LFS, that history sometimes needs rewriting, and that version control is part of engineering, not bookkeeping.

Automation, containerization, and deployment completed the education. CI/CD pipelines exposed hidden assumptions and enforced consistency. Docker clarified what reproducibility means and why production must be treated as a separate environment. Deploying to Hugging Face and Docker Hub taught me how authentication, file limits, and strict configurations work. Debugging logs, fixing mismatches, and refining workflows showed that system reliability depends on discipline, iteration, and attention to detail. Building ML systems is far more than training models it's about the infrastructure, automation, and engineering that make the model stable, accessible, and maintainable.

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
