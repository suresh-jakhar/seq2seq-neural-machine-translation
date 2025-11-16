---
title: Seq2Seq Neural Machine Translation
emoji: 🔤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Seq2Seq Neural Machine Translation

English to French translation using custom Transformer model.

**Live Demo:** [Try it here](https://sureshjakhar-seq2seq-neural-machine-translation.hf.space)

## API Usage

```bash
curl -X POST https://sureshjakhar-seq2seq-neural-machine-translation.hf.space/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how are you?"}'
```

## About

Custom trained Transformer model for English to French translation. Built with PyTorch and deployed on Hugging Face Spaces.

**Author:** Suresh Jakhar  
**Contact:** gettingsuresh@gmail.com  
**GitHub:** [@suresh-jakhar](https://github.com/suresh-jakhar)

**License:** Apache-2.0
