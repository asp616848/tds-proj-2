from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import uvicorn
import os
import zipfile
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import shutil

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the sentence transformer model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the question-answer database
# This could be stored in a JSON file, database, etc.
# For this example, we'll use a simple JSON file
with open('qa_database.json', 'r') as f:
    qa_database = json.load(f)

# Precompute embeddings for all questions in the database
question_embeddings = model.encode([item['question'] for item in qa_database])

@app.post("/api/")
async def answer_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    # Process the file if provided
    file_content = None
    if file:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Check if it's a zip file and extract if needed
            if file.filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Look for CSV files in the extracted content
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                if csv_files:
                    csv_path = os.path.join(temp_dir, csv_files[0])
                    df = pd.read_csv(csv_path)
                    if 'answer' in df.columns:
                        return {"answer": str(df['answer'].iloc[0])}
    
    # Encode the input question
    question_embedding = model.encode([question])
    
    # Calculate similarity with all questions in the database
    similarities = cosine_similarity(question_embedding, question_embeddings)[0]
    
    # Find the most similar question
    most_similar_idx = np.argmax(similarities)
    similarity_score = similarities[most_similar_idx]
    
    # Only return an answer if the similarity is above a threshold
    if similarity_score > 0.7:  # Adjust this threshold as needed
        return {"answer": qa_database[most_similar_idx]["answer"]}
    else:
        return {"answer": "I couldn't find a matching question in my database."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
