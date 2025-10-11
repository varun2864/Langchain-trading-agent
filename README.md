# LLM Web App

AI-powered financial assistant with stock trading, analysis, and document Q&A capabilities.

## Features

### Stock Analysis (Yahoo Finance)
- Real-time stock quotes for Indian stocks
- Technical indicators (EMA, RSI) with buy/sell signals
- Historical OHLCV data
- Company fundamentals and info
- Stock symbol search

### Trading (Zerodha Kite)
- Place orders
- View portfolio holdings
- Get GTT orders
- Real-time quotes

### Document Q&A
- Upload PDF documents
- Ask questions using RAG (FAISS vector search)
- Persistent document storage

## Tech Stack

- **Backend**: FastAPI, LangChain
- **LLM**: Grok-4 (via OpenRouter)
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers
- **APIs**: Zerodha Kite, Yahoo Finance

## API Endpoints

- `POST /chat` - Chat with AI agent
- `POST /upload` - Upload PDF documents
- `POST /clear` - Clear all documents
