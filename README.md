# BookLM - Intelligent Book Recommendation System

A sophisticated book recommendation system that combines the power of AI, vector similarity search, and natural language processing to provide personalized book recommendations.

## 🚀 Features

- **AI-Powered Recommendations**: Uses OpenAI's language models to provide intelligent, contextual book recommendations
- **Semantic Search**: Leverages HuggingFace embeddings and ChromaDB for similarity-based book discovery
- **Book Comparison**: Compare two books with AI-generated insights
- **Fast API**: Built with FastAPI for high-performance API endpoints
- **Vector Database**: ChromaDB for efficient similarity search and retrieval

https://github.com/user-attachments/assets/db395501-37e1-421c-952d-6c5ecf913e4a

## 🏗️ Architecture

The system consists of several key components:

1. **Data Layer**: SQLite database storing book metadata
2. **Vector Database**: ChromaDB for semantic similarity search
3. **AI Layer**: OpenAI LLM for intelligent recommendations
4. **API Layer**: FastAPI serving REST endpoints
5. **Frontend**: Static HTML/CSS/JS interface

Tools Used:

- OpenAI for providing the language models
- HuggingFace for embedding models
- ChromaDB for vector database functionality
- FastAPI for the web framework
- LangChain for AI/ML orchestration

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Sufficient disk space for book embeddings

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd BookLM
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the project root with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=your_openai_base_url_here (if needed)
LLM_MODEL=gemma-3-1b-it

# Embedding Model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Database Configuration
CSV_PATH=dataset/Best_books_ever[Cleaned].csv
DB_PATH=books.db
ROWS_LIMIT=100

# Vector Database
INDEX_PATH=chroma_books_index
```

### 4. Data Preparation

The original dataset I used for this project:
https://github.com/scostap/goodreads_bbe_dataset 

You can find a cleaned version of this dataset in the `dataset/` folder.

Ensure you have the book dataset in the `dataset/` folder. The system expects a CSV file with the following columns:
- `bookId`: Unique book identifier
- `title`: Book title
- `author`: Book author
- `rating`: Book rating
- `description`: Book description
- `genres`: Book genres
- `characters`: Book characters
- `coverImg`: Book cover image URL

## 🚀 Running the Application

### Start the Server

```bash
uvicorn main:app --reload
```

### First Run

On the first run, the system will:
1. Load book data from CSV into SQLite database
2. Create embeddings for book descriptions using HuggingFace
3. Store embeddings in ChromaDB for similarity search
4. Start the web server

This process may take a few minutes depending on the dataset size.

## 📖 Usage

### Web Interface

1. **Book Recommendations**: 
   - Navigate to the "Recommendation" tab
   - Enter your book preferences (e.g., "I want a fantasy book about magical worlds")
   - Get AI-powered recommendations with reasoning

2. **Book Comparison**:
   - Navigate to the "Compare" tab
   - Search book titles
   - Select two books
   - Get AI-generated comparison insights


## 🏗️ Project Structure

```
BookLM/
├── main.py             
├── requirements.txt     
├── README.md            
├── books.db             # SQLite database (auto-generated)
├── chroma_books_index/  # ChromaDB vector database (auto-generated)
├── books_1.Best_Books_Ever.csv 
├── dataset/            
│   ├── Best_books_ever[Cleaned].csv
│   └── dataset.ipynb
└── static/ 
    └── index.html
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_API_BASE`: Your OpenAI Base URL
- `LLM_MODEL`: OpenAI model to use (I used: gemma-3-1b-it)
- `EMBEDDING_MODEL_NAME`: HuggingFace embedding model
- `CSV_PATH`: Path to your book dataset CSV
- `DB_PATH`: SQLite database file path
- `ROWS_LIMIT`: Number of books to process (for testing)
- `INDEX_PATH`: ChromaDB index directory

### Performance Tuning

- **ROWS_LIMIT**: Reduce for faster initial setup, increase for more comprehensive recommendations
- **Chunk Size**: Modify `chunk_size` in `prepare_documents()` for different embedding granularity
- **Similarity Search**: Adjust `k` parameter in `query_db()` for more/fewer recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License

---

**Happy Reading! 📚✨** 
