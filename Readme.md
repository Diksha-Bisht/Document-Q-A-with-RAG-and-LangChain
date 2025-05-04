
# 🧾 Project: Document Q&A with RAG, LangChain, Pinecone & Hugging Face

## 🎯 **Goal**
Build a **Document Question Answering system** using **Retrieval-Augmented Generation (RAG)**. This system:
- Accepts `.pdf`, `.txt`, and `.docx` files
- Splits the document into chunks
- Generates embeddings for each chunk
- Stores them in **Pinecone vector database**
- Accepts user queries and answers them using a **Hugging Face LLM** (FLAN-T5) based on relevant chunks

---

## 🧰 **Tools & Libraries Used**

| Tool/Library       | Purpose                            |
|--------------------|------------------------------------|
| LangChain          | RAG pipeline, QA Chain             |
| Pinecone           | Vector database for storing chunks |
| Hugging Face       | LLM (FLAN-T5), Embeddings          |
| Transformers       | Model & tokenizer loading          |
| Sentence Transformers | Chunk embeddings                |
| Streamlit (optional) | For UI (if used)                |
| PyMuPDF / python-docx | PDF/Docx file parsing           |
| Google Colab       | Development environment            |

---

## 📋 **Prerequisites**
- Pinecone account with an active API key
- Hugging Face account (for using free models)
- Basic knowledge of Python and LangChain
- Google Colab with GPU enabled (for faster inference)

---

## 🔌 **Libraries Installed**
```bash
pip install -U langchain langchain-community langchain-huggingface langchain-pinecone
pip install transformers sentence-transformers pinecone-client pypdf python-docx
```

---

## 🪜 **Steps to Achieve This Project**

1. **Upload Document**  
   - Supports `.pdf`, `.txt`, `.docx`
   - Read and extract full text

2. **Chunk the Text**  
   - Split into overlapping chunks using `RecursiveCharacterTextSplitter`

3. **Generate Embeddings**  
   - Use `all-MiniLM-L6-v2` model via `HuggingFaceEmbeddings`

4. **Store Embeddings in Pinecone**  
   - Create or connect to Pinecone index
   - Upload vector embeddings of chunks

5. **Setup Retrieval-QA Chain**  
   - Use Hugging Face model (`google/flan-t5-base`) via LangChain's `HuggingFacePipeline`

6. **Query & Answer**  
   - Accept user input
   - Retrieve relevant chunks
   - Generate and return an answer using LLM
   - Loop until user types `'Q'` to quit

---
## 📋 **Definations**
1. **RAG (Retrieval-Augmented Generation)** 
   - **Retrieval (R):** The system retrieves relevant chunks from the document (stored in Pinecone) based on the user's query. These chunks are the "knowledge" that will be used to answer the query.
   - **Augmented (A):** The retrieved chunks are used to augment the context or information for the model (the Hugging Face LLM, such as FLAN-T5). This gives the model more context about the user's query, making it more informed and capable of generating relevant answers.
   - **Generation (G):** The augmented context (relevant chunks) is passed to the Hugging Face model for text generation, producing an answer to the user’s query based on the retrieved content.

2. **LangChain**
   - A framework for building language model (LLM)-based applications, including workflows like Retrieval-Augmented Generation (RAG). LangChain allows the easy integration of external tools, such as Pinecone (for vector storage) and Hugging Face models, to construct powerful pipelines for natural language understanding and generation.
     
3. **Pinecone**
   - A vector database used to store and search high-dimensional vector embeddings. In this project, Pinecone is used to store the embeddings of document chunks, enabling efficient similarity-based retrieval. Pinecone enables fast and scalable search in large datasets by converting documents into vectors and matching them with query embeddings.

4. **Document Chunking**
   - The process of dividing a large document into smaller, more manageable sections (chunks). This step is important for efficient retrieval and processing, as it allows the system to focus on smaller, relevant parts of the document rather than the entire text. Chunking is typically done based on a character length or semantic similarity to ensure the chunks contain meaningful content.

5. **Embeddings**
   - A mathematical representation of text (such as a word, sentence, or document) in a high-dimensional vector space. In this project, the embedding model (e.g., all-MiniLM-L6-v2) is used to transform the document chunks and user query into fixed-length vectors. These vectors are then stored in Pinecone for similarity search.

6. **Hugging Face**
   - An open-source platform and library that provides pre-trained models for natural language processing (NLP). In this project, a Hugging Face transformer model (such as FLAN-T5) is used for generating responses to user queries. Hugging Face models are popular for tasks such as text generation, question answering, translation, and more.

7. **Vector Database**
   - A database that stores vector representations (embeddings) of data, typically used for similarity search. Pinecone is an example of a vector database that facilitates efficient nearest-neighbor search for high-dimensional vectors. It is useful for tasks like document retrieval in RAG systems.

8. **LLM (Large Language Model)**
   - A type of neural network trained on vast amounts of text data to generate human-like language. FLAN-T5 is an example of an LLM used in this project to generate answers to user queries based on context retrieved from Pinecone.

9. **Retrieval-QA**
   - A task in natural language processing where a system is designed to retrieve relevant information from a corpus or database and use it to answer a user’s question. In this project, LangChain facilitates this process by integrating document retrieval with question answering using a generative model.

10. **Text Embedding**
   - The process of converting text (a word, sentence, or document) into a vector of numbers. This transformation is performed using models like Sentence-BERT or other transformers to capture semantic meaning in a high-dimensional space. These embeddings enable efficient matching of similar text segments during retrieval tasks.
    
11. **RecursiveCharacterTextSplitter**
   - A text splitter used in LangChain to split long documents into smaller chunks. It does so by recursively splitting the text based on a character length threshold, ensuring that each chunk is manageable and maintains context for semantic understanding.
