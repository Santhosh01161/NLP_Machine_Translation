# 🌏 English to Tamil Neural Machine Translation (NMT) 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge&logo=flask)

> **A Transformer-based Sequence-to-Sequence model built from scratch to translate English text into Tamil.**

---

## 📖 Project Overview

This project implements a **Neural Machine Translation (NMT)** system using the **Transformer architecture**. The goal was to build a model capable of understanding English sentence structures and generating accurate Tamil translations.

The core of the assignment involves comparing different **Attention Mechanisms** to analyze their impact on translation quality and model convergence. The final deployed model in this repository uses **Additive Attention**, which showed the lowest training loss.

### ✨ Key Features
* **🧠 Transformer Architecture:** Implemented from scratch (Encoder-Decoder, Multi-Head Attention).
* **🔍 Custom Tokenizer:** Uses `indic-nlp-library` for Tamil and standard tokenization for English.
* **🎨 Web Interface:** A clean Flask-based UI to test translations in real-time.
* **📊 Visualization:** Includes attention maps to visualize how the model aligns words between languages.

---

## 🏆 Performance & Results

I experimented with different attention mechanisms to optimize performance. Below is the comparative analysis of the models trained.

| Attention Variant | Training Loss | Training PPL | Validation Loss | Validation PPL | BLEU Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🔴 **General** | 2.143 | 8.523 | 5.341 | 208.714 | 100 | 63.5m |
| 🟢 **Additive** | **1.934** | **6.92** | 5.411 | 223.805 | 0 | 85.0m |

> **Observation:** The **Additive** model achieved the best (lowest) Training Loss and Perplexity (PPL), indicating it learned the training data effectively, though it required more training time than the General attention mechanism.
<img width="1106" height="830" alt="Screenshot 2026-02-07 at 7 49 15 pm" src="https://github.com/user-attachments/assets/57d3d391-0468-4460-b1cc-889f1a593918" />
<img width="1106" height="830" alt="Screenshot 2026-02-07 at 7 53 13 pm" src="https://github.com/user-attachments/assets/75b012fe-b22a-4a39-ae9c-16cba8134a61" />


---

## 📂 Repository Structure

This repository is organized to separate the model training logic from the web deployment.

```bash
📦 NLP_Machine_Translation
 ┣ 📜 A3_Machine_Tranalation (1).ipynb  # 📓 Jupyter Notebook (Training Code & Analysis)
 ┣ 📜 en-ta-transformer-additive.pt     # 🧠 Model Weights (Additive Attention)
 ┣ 📜 en_vocab.pth                      # 📖 English Vocabulary Mapping
 ┣ 📜 ta_vocab.pth                      # 📖 Tamil Vocabulary Mapping
 ┣ 📜 .gitignore                        # 🚫 Git Ignore File
 ┣ 📜 README.md                         # 📄 Project Documentation
 ┗ 📂 app
 ┃ ┣ 📜 app.py                          # 🚀 Flask Application (Inference Logic)
 ┃ ┣ 📜 requirements.txt                # 📦 Dependencies
 ┃ ┗ 📂 templates
 ┃ ┃ ┗ 📜 index.html                    # 🎨 Web Interface (HTML/CSS)
