from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
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
with open('qa_database.json', 'r') as f:
    qa_database = json.load(f)

# Precompute embeddings for all questions in the database
question_embeddings = model.encode([item['question'] for item in qa_database])

@app.post("/api/")
async def answer_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    file_content = None
    if file:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            if file.filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
                if csv_files:
                    csv_path = os.path.join(temp_dir, csv_files[0])
                    df = pd.read_csv(csv_path)
                    if 'answer' in df.columns:
                        return {"answer": str(df['answer'].iloc[0])}

    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, question_embeddings)[0]
    most_similar_idx = np.argmax(similarities)
    similarity_score = similarities[most_similar_idx]

    if similarity_score > 0.7:
        return {"answer": qa_database[most_similar_idx]["answer"]}
    else:
        return {"answer": "I couldn't find a matching question in my database."}
