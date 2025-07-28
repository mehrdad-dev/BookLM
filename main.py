"""
This application provides an intelligent book recommendation system using:
- LangChain for AI/ML operations
- ChromaDB for vector similarity search
- FastAPI for the web API
- SQLite for book data storage
- HuggingFace embeddings for semantic search

The system works by:
1. Loading book data from CSV into SQLite
2. Creating embeddings for book descriptions
3. Storing embeddings in ChromaDB for similarity search
4. Using LLM to provide intelligent recommendations
"""

# libraries imports
from dotenv import load_dotenv
import pandas as pd
import warnings
import sqlite3
import json
import os


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI


from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI


warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME") 
index_path = os.getenv("INDEX_PATH")
rows_limit = int(os.getenv("ROWS_LIMIT"))

# --- AI Prompt Template Setup ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You will be given a user query and a list of matched books to recommend the best book based on the user query. 
Return the title of the best book and the reason for the recommendation in a json format.
Output format: {{"title": "title of the best book", "reasoning": "reason for the recommendation"}}
Your reasoning must be in a friendly and engaging tone, and you should speak directly to the user.
"""),
        ("human", "User Query: {query}\n Matched Books: {matched_books}"),  
    ]
)

# Initialize the LLM
llm = ChatOpenAI(model=os.getenv("LLM_MODEL"), temperature=0)

# --- Embedding Model Setup ---
print('🔁 Loading embedding model ...\n')
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name)

# --- Database Functions ---

def load_csv_to_sqlite(csv_path, db_path, table_name="books", nrows=None):
    """
    Load CSV book data into SQLite database.
    
    Args:
        csv_path (str): Path to the CSV file containing book data
        db_path (str): Path to the SQLite database file
        table_name (str): Name of the table to create
        nrows (int): Number of rows to load (for testing with smaller datasets)
    
    Returns:
        sqlite3.Connection: Database connection
    """
    conn = sqlite3.connect(db_path)
    # Check if table exists to avoid recreating it
    cur = conn.cursor()
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    exists = cur.fetchone()
    if not exists:
        df = pd.read_csv(csv_path)
        if nrows:
            df = df.head(nrows)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    return conn

def get_book_by_id(book_id, conn, table_name="books"):
    """
    Retrieve a book by its unique ID.
    
    Args:
        book_id (str): The unique book identifier
        conn: SQLite database connection
        table_name (str): Name of the books table
    
    Returns:
        dict: Book data as dictionary, or None if not found
    """
    query = f"SELECT * FROM {table_name} WHERE bookId = ?"
    cur = conn.cursor()
    cur.execute(query, (book_id,))
    row = cur.fetchone()
    if row:
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))
    else:
        return None

def get_book_by_title(book_title, conn, table_name="books"):
    """
    Search for books by title (case-insensitive partial match).
    
    Args:
        book_title (str): Title to search for
        conn: SQLite database connection
        table_name (str): Name of the books table
    
    Returns:
        dict: Dictionary with 'results' key containing list of matching books
    """
    query = f"SELECT title, author, rating, description, coverImg FROM {table_name} WHERE LOWER(title) LIKE ? LIMIT 20"
    with sqlite3.connect("books.db") as conn:
        cur = conn.cursor()
        cur.execute(query, (f"%{book_title.lower()}%",))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        results = [dict(zip(columns, row)) for row in rows]
    return {"results": results}

# --- Document Processing Functions ---

def create_doc(row):
    """
    Create a formatted document string from book data for embedding.
    
    Args:
        row: Pandas DataFrame row containing book data
    
    Returns:
        str: Formatted book information
    """
    return f"""Title: {row['title']}
                Author: {row['author']}
                Rating: {row['rating']}
                Genres: {row['genres']}
                Characters: {row['characters']}
                Description: {row['description']}"""

def prepare_documents(limit=100):
    """
    Prepare book documents for vector database indexing.
    
    Args:
        limit (int): Maximum number of books to process
    
    Returns:
        list: List of Document objects ready for embedding
    """
    conn = sqlite3.connect("books.db")
    df = pd.read_sql_query(f"SELECT * FROM books LIMIT {limit}", conn)
    conn.close()

    df["doc"] = df.apply(create_doc, axis=1)

    documents = [
        Document(
            page_content=str(row["doc"]),
            metadata={
                "bookId": row["bookId"],
            }
        )
        for _, row in df.iterrows()
    ]

    rec_char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100)
    
    return rec_char_splitter.split_documents(documents)

def create_vectordb(index_path, embedding_model):
    """
    Create or load the ChromaDB vector database for similarity search.
    
    Args:
        index_path (str): Path to store/load the vector database
        embedding_model: HuggingFace embeddings model
    
    Returns:
        Chroma: Vector database instance
    """
    print('='*50)
    if os.path.exists(index_path):
        print("🔁 Chroma index found. Loading existing index...")
        db = Chroma(persist_directory=index_path, embedding_function=embedding_model)
    else:
        print("⚙️ No index found. Creating new one...")
        print('🔁 Preparing documents...')
        docs = prepare_documents(limit=rows_limit)
        print(f"🔁 Creating Chroma index...")
        db = Chroma.from_documents(docs, embedding_model, persist_directory=index_path)
        print("✅ New Chroma index saved to {index_path}.")

    return db

# --- Data Loading ---
csv_path = os.getenv("CSV_PATH")
db_path = os.getenv("DB_PATH")
conn = load_csv_to_sqlite(csv_path, db_path, table_name="books", nrows=rows_limit)

# Initialize the vector database
db = create_vectordb(index_path=index_path, embedding_model=embedding_model)

# --- Recommendation Functions ---

def query_db(db, query, k=5):
    """
    Search the vector database for similar books.
    
    Args:
        db: ChromaDB instance
        query (str): User's search query
        k (int): Number of similar books to return
    
    Returns:
        list: List of similar documents
    """
    return db.similarity_search(query, k=k)

def get_recommendations(query, k=5):
    """
    Get book recommendations based on user query.
    
    Args:
        query (str): User's book preference query
        k (int): Number of recommendations to return
    
    Returns:
        list: List of recommended books with full details
    """
    # Get similar books from vector database
    results = query_db(db, query, k=k)
    recs = []
    
    with sqlite3.connect("books.db") as conn:
        for doc in results:
            book_id = doc.metadata.get("bookId")
            book_data = get_book_by_id(book_id, conn, table_name="books")
            if book_data:
                recs.append({
                    "title": book_data.get("title", "N/A"),
                    "description": book_data.get("description", "N/A"),
                    "rating": book_data.get("rating", "N/A"),
                    "genres": book_data.get("genres", "N/A"),
                    "author": book_data.get("author", "N/A"),
                    "coverImg": book_data.get("coverImg", "N/A"),
                })

    # Sort by rating descending
    recs.sort(key=lambda x: float(x["rating"]) if str(x["rating"]).replace('.','',1).isdigit() else 0, reverse=True)
    return recs

# --- FastAPI Application Setup ---

app = FastAPI(title="BookLM", 
              description="An intelligent book recommendation system")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/recommend")
def recommend(query: str = "I want to read a book fantasy book which is about magical worlds."):
    """
    Get intelligent book recommendations based on user query.
    
    Args:
        query (str): User's book preference description
    
    Returns:
        dict: Contains LLM reasoning and list of recommendations
    """
    recs = get_recommendations(query)
    
    prompt = prompt_template.format(query=query, matched_books=recs)
    response = llm.invoke(prompt)
    
    # Parse the LLM's JSON output
    try:
        content_str = response.content if isinstance(response.content, str) else str(response.content)
        # Remove markdown code block and json tag if present
        lines = content_str.strip().splitlines()
        lines = [line for line in lines if not line.strip().startswith('```')]
        content_str = '\n'.join(lines).strip()
        llm_json = json.loads(content_str)
        best_title = llm_json.get("title", "Unknown")
        reasoning = llm_json.get("reasoning", "")
        llm_result = {"title": best_title, "reasoning": reasoning}
    except Exception as e:
        llm_result = {"title": "Error", "reasoning": f"Could not parse LLM output: {e}\nRaw output: {response.content}"}
    
    return {"llm_result": llm_result, "recommendations": recs}

@app.get("/compare")
def compare(query1: str, query2: str = ""):
    """
    Compare two books by title and provide AI-generated comparison.
    
    Args:
        query1 (str): First book title to search for
        query2 (str): Second book title to search for (optional)
    
    Returns:
        dict: Book comparison results or search results
    """
    results1 = get_book_by_title(query1, conn, table_name="books")
    results2 = get_book_by_title(query2, conn, table_name="books") if query2 else {"results": []}

    # If both books are found, generate AI comparison
    if query1 and query2 and results1.get("results") and results2.get("results"):
        book1_info = str(results1["results"][0]["title"]) + "\n" + str(results1["results"][0]["description"]) + "\n" + str(results1["results"][0]["rating"]) + "\n" + str(results1["results"][0]["author"])
        book2_info = str(results2["results"][0]["title"]) + "\n" + str(results2["results"][0]["description"]) + "\n" + str(results2["results"][0]["rating"]) + "\n" + str(results2["results"][0]["author"])
        
        compare_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that compares two books for a user. Given the details of two books, provide a friendly, engaging, and insightful comparison. Mention strengths, weaknesses, and which type of reader might prefer each. Output should be a paragraph, not a table."),
            ("human", "Book 1: {book1_info}\nBook 2: {book2_info}")
        ])
        
        response = llm.invoke(compare_prompt.format(book1_info=str(book1_info), book2_info=str(book2_info)))
        content_str = response.content if isinstance(response.content, str) else str(response.content)
        return {"comparison": content_str}

    return {"results": {"results1": results1.get("results", [])[:5], "results2": results2.get("results", [])[:5]}}

# HTML frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")