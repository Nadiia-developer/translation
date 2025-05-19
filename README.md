This project is a bite-sized demonstration of:

* Building an encoder-decoder (seq-to-seq) model in TensorFlow /Keras  
* Adding a simple **self-attention** layer to both encoder and decoder  
* Comparing two initialiser choices — **Glorot (uniform)** vs **He (uniform)** — while keeping the optimiser (Adam) fixed  
* Visualising the loss curves for a very small toy dataset (English → Spanish sentences)

Although the dataset here is deliberately tiny, the code skeleton is ready for larger parallel corpora.

---

## File overview

| File        | Purpose |
|-------------|---------|
| `main.py`   | Main script that tokenises data, builds the models, trains twice (different initialisers) and plots the loss curves. |
| `README.md` | What you are reading now. |

---

## Quick start

```bash
# 1. Clone or download this repo
git clone https://github.com/Nadiia-developer/translation.git

# 2. Сreate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install requirements
pip install -r requirements.txt
# or, if you don’t keep a requirements.txt, install manually:
pip install tensorflow matplotlib numpy

# 4. Run the script
python main.py
