# pip install streamlit PyPDF2 langchain faiss-cpu sentence-transformers
# tokens hf = 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


import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import os

def main():
    st.title("PDF QA with Free LLM")
    st.write("Upload PDFs, then ask questions about their content")

    # Sidebar for API key and model selection
    with st.sidebar:
        st.header("Configuration")
        hf_token = st.text_input("HuggingFace Hub API Token", type="password")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = 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
        
        model_options = {
            "google/flan-t5-xl": "Large but accurate",
            "facebook/bart-large-cnn": "Good for summarization",
            "declare-lab/flan-alpaca-large": "Instruction-tuned model",
            "bigscience/bloom-7b1": "Large multilingual model"
        }
        selected_model = st.selectbox(
            "Choose a model",
            options=list(model_options.keys()),
            format_func=lambda x: f"{x} ({model_options[x]})"
        )

    # PDF upload section
    st.header("Upload PDFs")
    pdf_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Process PDFs button
    process_pdfs = st.button("Process PDFs")

    # Initialize session state for vector store
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if process_pdfs and pdf_files:
        with st.spinner("Processing PDFs..."):
            # Extract text from PDFs
            text = ""
            for pdf_file in pdf_files:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""

            # Split text into chunks with fixed values
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            
            st.success(f"Processed {len(pdf_files)} PDF(s) with {len(chunks)} text chunks!")

    # QA section
    st.header("Ask Questions")
    question = st.text_input("Enter your question about the PDF content")

    if st.button("Get Answer") and question:
        if not st.session_state.vector_store:
            st.warning("Please upload and process PDFs first")
        elif not hf_token:
            st.warning("Please enter your HuggingFace Hub API token")
        else:
            with st.spinner("Searching for answer..."):
                try:
                    # Perform similarity search
                    docs = st.session_state.vector_store.similarity_search(question, k=3)
                    
                    # Load QA chain with selected model
                    llm = HuggingFaceHub(
                        repo_id=selected_model,
                        model_kwargs={
                            "temperature": 0.1,
                            "max_length": 512
                        },
                        task="text-generation"  # Explicitly set the task
                    )
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    # Get answer
                    response = chain.run(input_documents=docs, question=question)
                    
                    st.subheader("Answer:")
                    st.write(response)
                    
                    # Show source chunks
                    with st.expander("See relevant text chunks"):
                        for i, doc in enumerate(docs):
                            st.write(f"Chunk {i+1}:")
                            st.text(doc.page_content)
                            st.divider()
                            
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.info("Try selecting a different model from the sidebar")

if __name__ == "__main__":
    main()
