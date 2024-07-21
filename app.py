import streamlit as st
from htmlTemplates import css, bot_template, user_template
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from PyPDF2 import PdfReader
from pptx import Presentation
from docx import Document
import tempfile

model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page_num, page in enumerate(pdf_reader.pages, 1):
        text += f"Page {page_num}:\n"
        text += page.extract_text()
        text += "\n\n"  # Add space between pages
    return text

def extract_text_from_pptx(pptx_file):
    presentation = Presentation(pptx_file)
    text = ""
    for slide_num, slide in enumerate(presentation.slides, 1):
        text += f"Slide {slide_num}:\n"
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text += f"{paragraph.text}\n"
        text += "\n"  # Add a separator between slides
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        if para.style.name.startswith('Heading'):
            text += f"\n\n{para.text}\n\n"  # Add extra space around headings
        else:
            text += para.text + " "  # Combine lines into paragraphs

    # Ensure paragraphs are separated properly
    paragraphs = text.split("\n\n")
    formatted_text = "\n\n".join(para.strip() for para in paragraphs if para.strip())

    for table in doc.tables:
        formatted_text += "\n\n"
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    formatted_text += cell.text + "\t"
            formatted_text += "\n"
        formatted_text += "\n"

    return formatted_text

def get_response(question):
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=3,
        google_api_key="AIzaSyByAULS9YrPUqkrai_ZQn5-PPaOxBaTpcU"  
    )

    retriever = db.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    response = chain.invoke({"input": question})
    return response['answer']

def handle_userinput(user_question):
    response = get_response(user_question)
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Convo AI: Chat with multiple documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Converse with your documents with Convo AI :books:")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload your documents (PDFs, PPTs, or DOCXs) and click on 'Analyze Documents'", accept_multiple_files=True, type=['pdf', 'pptx', 'docx'])
        
        if st.button("Analyze Documents"):
            with st.spinner("Processing your Documents..."):
                if uploaded_files:
                    text_final = ""
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name[-5:]) as temp_file:
                            temp_file.write(file.getvalue())
                            temp_file_path = temp_file.name

                        if file.name.endswith('.pdf'):
                            text = get_pdf_text(temp_file_path).strip()
                        elif file.name.endswith('.pptx'):
                            text = extract_text_from_pptx(temp_file_path).strip()
                        elif file.name.endswith('.docx'):
                            text = extract_text_from_docx(temp_file_path).strip()
                        else:
                            text = ""

                        text_final += text
                        os.unlink(temp_file_path)  # Delete the temporary file

                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
                    texts = text_splitter.split_text(text_final)

                    db = FAISS.from_texts(texts, embeddings)
                    db.save_local("vectorstore")

                    st.success("Documents processed successfully!")
                else:
                    st.warning("Please upload at least one file.")

if __name__ == "__main__":
    main()
