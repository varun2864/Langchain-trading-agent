import os
import json
import shutil
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from kiteconnect import KiteConnect
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

try:
    import yfinance as yf
except ImportError:
    yf = None

# Load environment variables first
load_dotenv()

# Get API keys from environment
kite_api_key = os.getenv("KITE_API_KEY")
kite_api_secret = os.getenv("KITE_API_SECRET")

# --- Helper Functions ---

def check_yfinance() -> str:
    """Check if yfinance is available and return error message if not."""
    if yf is None:
        return "Error: yfinance package not installed. Run: pip install yfinance"
    return None

# --- KiteConnect Tool Functions ---

kite = None  # Global instance for simplicity

def configure_kite(request_token: str) -> str:
    """Configure KiteConnect using request token (API key and secret loaded from env)."""
    global kite
    if not kite_api_key or not kite_api_secret:
        return "Error: KITE_API_KEY and KITE_API_SECRET must be set in .env file"
    
    kite = KiteConnect(api_key=kite_api_key)
    try:
        data = kite.generate_session(request_token, api_secret=kite_api_secret)
        kite.set_access_token(data["access_token"])
        profile = kite.profile()
        return f"Connected: {profile.get('user_name')}"
    except Exception as e:
        return f"Error during configuration: {e}"

def get_quote(symbols: List[str]) -> str:
    if not kite:
        return "Configure first"
    try:
        quotes = kite.quote(symbols)
        return f"Quotes: {quotes}"
    except Exception as e:
        return f"Error fetching quotes: {e}"

def place_order(order: str) -> str:
    """Place an order. Input should be a JSON string or dict with symbol, exchange, transaction_type, quantity, product, order_type, and optional price."""
    if not kite:
        return "Configure first"
    try:
        # Parse if string, otherwise use as-is
        if isinstance(order, str):
            order_dict = json.loads(order.replace("'", '"'))
        else:
            order_dict = order
            
        order_id = kite.place_order(
            tradingsymbol=order_dict["symbol"],
            exchange=order_dict["exchange"],
            transaction_type=order_dict["transaction_type"],
            quantity=order_dict["quantity"],
            product=order_dict["product"],
            order_type=order_dict["order_type"],
            price=order_dict.get("price"),
            variety="regular"
        )
        return f"Order placed: {order_id}"
    except Exception as e:
        return f"Error placing order: {e}"

def get_holdings(dummy: str = "") -> str:
    """Get portfolio holdings. No input required."""
    if not kite:
        return "Configure first"
    try:
        holdings = kite.holdings()
        return f"Holdings: {holdings}"
    except Exception as e:
        return f"Error fetching holdings: {e}"

def get_gtt_orders(dummy: str = "") -> str:
    """Get GTT orders. No input required."""
    if not kite:
        return "Configure first"
    try:
        gtt_orders = kite.gtts()
        return f"GTT Orders: {gtt_orders}"
    except Exception as e:
        return f"Error fetching GTT orders: {e}"

# --- Yahoo Finance Tool Functions ---

def get_stock_quote_yf(symbol: str) -> str:
    """Get real-time stock quote using Yahoo Finance. For NSE stocks use .NS suffix (e.g., RELIANCE.NS), for BSE use .BO suffix."""
    error = check_yfinance()
    if error:
        return error
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'symbol' not in info:
            return f"No data found for symbol: {symbol}. For Indian stocks, use .NS (NSE) or .BO (BSE) suffix (e.g., RELIANCE.NS)"
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 'N/A')
        change = info.get('regularMarketChange', 'N/A')
        change_percent = info.get('regularMarketChangePercent', 'N/A')
        volume = info.get('volume', 'N/A')
        
        return f"Symbol: {symbol}, Price: ₹{current_price}, Change: {change} ({change_percent}%), Volume: {volume}"
    except Exception as e:
        return f"Error fetching stock quote: {e}"

def search_indian_stock(keywords: str) -> str:
    """Search for Indian stock symbols by company name. Returns NSE symbols with .NS suffix."""
    error = check_yfinance()
    if error:
        return error
    
    try:
        # Common Indian stock symbols mapping (you can expand this)
        indian_stocks = {
            'reliance': 'RELIANCE.NS', 'tcs': 'TCS.NS', 'hdfc': 'HDFCBANK.NS',
            'infosys': 'INFY.NS', 'icici': 'ICICIBANK.NS', 'bharti': 'BHARTIARTL.NS',
            'itc': 'ITC.NS', 'sbi': 'SBIN.NS', 'bajaj': 'BAJFINANCE.NS',
            'asian paints': 'ASIANPAINT.NS', 'hul': 'HINDUNILVR.NS', 'maruti': 'MARUTI.NS'
        }
        
        keywords_lower = keywords.lower()
        matches = []
        for name, symbol in indian_stocks.items():
            if keywords_lower in name:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                company_name = info.get('longName', name.title())
                matches.append(f"{symbol} - {company_name}")
        
        if matches:
            return "Search Results (use .NS for NSE, .BO for BSE):\n" + "\n".join(matches)
        else:
            return f"No matches found. Try using the exact NSE symbol with .NS suffix (e.g., RELIANCE.NS) or BSE with .BO suffix"
    except Exception as e:
        return f"Error searching symbols: {e}"

def get_company_info_yf(symbol: str) -> str:
    """Get company information and fundamental data using Yahoo Finance."""
    error = check_yfinance()
    if error:
        return error
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'symbol' not in info:
            return f"No data found for: {symbol}"
        
        result = f"Company: {info.get('longName', 'N/A')}\n"
        result += f"Symbol: {symbol}\n"
        result += f"Sector: {info.get('sector', 'N/A')}\n"
        result += f"Industry: {info.get('industry', 'N/A')}\n"
        result += f"Market Cap: ₹{info.get('marketCap', 'N/A')}\n"
        result += f"P/E Ratio: {info.get('trailingPE', 'N/A')}\n"
        result += f"52 Week High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}\n"
        result += f"52 Week Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}\n"
        result += f"Dividend Yield: {info.get('dividendYield', 'N/A')}\n"
        result += f"Description: {info.get('longBusinessSummary', 'N/A')[:300]}..."
        
        return result
    except Exception as e:
        return f"Error fetching company info: {e}"

def get_historical_data(symbol: str, period: str = "1mo") -> str:
    """Get historical OHLCV data. Period options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max. Default: 1mo"""
    error = check_yfinance()
    if error:
        return error
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            return f"No historical data found for: {symbol}"
        
        # Return last 10 rows as a formatted string
        result = f"Historical data for {symbol} (Period: {period}, Last 10 rows):\n\n"
        result += hist.tail(10).to_string()
        return result
    except Exception as e:
        return f"Error fetching historical data: {e}"

def calculate_technical_indicators(symbol: str, ema_period: int = 20, rsi_period: int = 14, data_period: str = "3mo") -> str:
    """Calculate EMA and RSI technical indicators. Default: 20-day EMA, 14-day RSI over 3 months of data."""
    error = check_yfinance()
    if error:
        return error
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=data_period)
        
        if hist.empty:
            return f"No data found for: {symbol}"
        
        current_price = hist['Close'].iloc[-1]
        
        # Calculate EMA
        ema = hist['Close'].ewm(span=ema_period, adjust=False).mean()
        current_ema = ema.iloc[-1]
        
        # Calculate RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Build result
        result = f"Technical Analysis for {symbol}:\n\n"
        result += f"Current Price: ₹{current_price:.2f}\n\n"
        
        result += f"EMA-{ema_period}: ₹{current_ema:.2f}\n"
        if current_price > current_ema:
            result += "EMA Signal: Price ABOVE EMA (Bullish)\n\n"
        else:
            result += "EMA Signal: Price BELOW EMA (Bearish)\n\n"
        
        result += f"RSI-{rsi_period}: {current_rsi:.2f}\n"
        if current_rsi > 70:
            result += "RSI Signal: OVERBOUGHT (Consider Selling)\n\n"
        elif current_rsi < 30:
            result += "RSI Signal: OVERSOLD (Consider Buying)\n\n"
        else:
            result += "RSI Signal: NEUTRAL\n\n"
        
        # Overall recommendation
        if current_price > current_ema and current_rsi < 70:
            result += "Overall: BULLISH trend with room to grow"
        elif current_price < current_ema and current_rsi > 30:
            result += "Overall: BEARISH trend with room to fall"
        elif current_rsi > 70:
            result += "Overall: OVERBOUGHT - caution advised"
        elif current_rsi < 30:
            result += "Overall: OVERSOLD - potential buying opportunity"
        else:
            result += "Overall: NEUTRAL - mixed signals"
        
        return result
    except Exception as e:
        return f"Error calculating technical indicators: {e}"

# --- RAG Setup ---

vectorstore = None
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Try to load existing FAISS index at startup
try:
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings_model, allow_dangerous_deserialization=True
    )
    print("Loaded existing FAISS index")
except:
    print("No existing FAISS index found, will create on first upload")

# --- RAG Tool Function ---

def retrieve_from_documents(query: str) -> str:
    """
    Retrieve relevant context from uploaded documents using FAISS and return as a string.
    """
    global vectorstore
    if not vectorstore:
        return "No documents uploaded yet."
    try:
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return "No relevant information found in uploaded documents."
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        return f"Error retrieving from documents: {e}"

# --- Tool Wrapping ---

kite_tools = [
    Tool.from_function(
        func=configure_kite,
        name="configure_kite",
        description="Tool name: configure_kite (use exact name without escaping). Configure KiteConnect using request_token string (API key and secret loaded from environment variables)."
    ),
    Tool.from_function(
        func=get_quote,
        name="get_quote",
        description="Tool name: get_quote (use exact name without escaping). Get quotes for a list of symbols. Input should be a list of symbol strings."
    ),
    Tool.from_function(
        func=place_order,
        name="place_order",
        description="Tool name: place_order (use exact name without escaping). Place an order. Requires a dict with symbol, exchange, transaction_type, quantity, product, order_type, and optional price."
    ),
    Tool.from_function(
        func=get_holdings,
        name="get_holdings",
        description="Tool name: get_holdings (use exact name without escaping). Get portfolio holdings. No input required - just call this tool with empty string."
    ),
    Tool.from_function(
        func=get_gtt_orders,
        name="get_gtt_orders",
        description="Tool name: get_gtt_orders (use exact name without escaping). Get GTT orders. No input required - just call this tool with empty string."
    ),
]

rag_tool = Tool.from_function(
    func=retrieve_from_documents,
    name="retrieve_from_documents",
    description="Tool name: retrieve_from_documents (use exact name without escaping). Retrieve relevant information from uploaded documents given a user query string."
)

yahoo_finance_tools = [
    Tool.from_function(
        func=get_stock_quote_yf,
        name="get_stock_quote_yf",
        description="Tool name: get_stock_quote_yf (use exact name without escaping). Get real-time stock quote using Yahoo Finance. For Indian NSE stocks use .NS suffix (e.g., 'RELIANCE.NS'), for BSE use .BO suffix (e.g., 'RELIANCE.BO'). Input: stock symbol string"
    ),
    Tool.from_function(
        func=search_indian_stock,
        name="search_indian_stock",
        description="Tool name: search_indian_stock (use exact name without escaping). Search for Indian stock symbols by company name. Returns NSE symbols with .NS suffix. Input: company name or keywords string"
    ),
    Tool.from_function(
        func=get_company_info_yf,
        name="get_company_info_yf",
        description="Tool name: get_company_info_yf (use exact name without escaping). Get company information and fundamental data using Yahoo Finance. Input: stock symbol string (e.g., 'RELIANCE.NS')"
    ),
    Tool.from_function(
        func=get_historical_data,
        name="get_historical_data",
        description="Tool name: get_historical_data (use exact name without escaping). Get historical OHLCV (Open, High, Low, Close, Volume) data. Input: symbol and optional period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max). Default period: 1mo"
    ),
    Tool.from_function(
        func=calculate_technical_indicators,
        name="calculate_technical_indicators",
        description="Tool name: calculate_technical_indicators (use exact name without escaping). Calculate EMA and RSI technical indicators with overall analysis. Input: symbol, optional ema_period (default 20), optional rsi_period (default 14), optional data_period (default 3mo). Returns both indicators with buy/sell signals and overall recommendation."
    )
]

all_tools = kite_tools + [rag_tool] + yahoo_finance_tools

# --- LLM and Agent Setup ---

app = FastAPI()

origins = [
    "http://localhost",
    "http://127.0.0.1:8000",
    "file:///",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("OPENROUTER_API_KEY")

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=api_key,
    model_name="x-ai/grok-4-fast"
)

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=all_tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    message: str

# --- FastAPI Endpoints ---

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Accepts a message in JSON body and returns the agent's response.
    """
    try:
        response = agent.run(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF file, splits it, creates embeddings, and updates the FAISS index.
    """
    global vectorstore
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        if vectorstore:
            vectorstore.add_documents(docs)
        else:
            vectorstore = FAISS.from_documents(docs, embeddings_model)

        vectorstore.save_local("faiss_index")
        return {"message": f"Document '{file.filename}' uploaded and processed successfully."}
    except Exception as e:
        print(f"Upload error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/clear")
def clear_documents():
    """
    Clear all documents from the FAISS index.
    """
    global vectorstore
    try:
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
        vectorstore = None
        return {"message": "All documents cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
