# AI Text Summarization Web App

## Overview
An intelligent text summarization application powered by AI, capable of generating concise summaries from input text using state-of-the-art NLP techniques.

## 🚀 Features
- AI-powered text summarization
- Interactive web interface
- Quick and efficient content condensation
- Supports various text lengths

## 🛠 Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Gradio
- DistilBART Model

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual Environment (recommended)

### Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/text-summarization.git
cd text-summarization
```

2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🖥 Usage
```bash
python textsummary.py
```

## 🔍 How It Works
The application uses the DistilBART model from Hugging Face Transformers to generate summaries. It provides a simple Gradio interface where users can input text and receive a concise summary.
