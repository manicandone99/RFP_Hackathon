# import streamlit as st
# import os
# import pickle
# import pandas as pd
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
# from langchain_community.document_loaders import  PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from dotenv import load_dotenv
# import time
# import io


# # Load environment variables
# load_dotenv()
# nvidia_api_key = os.getenv("NVIDIA_API_KEY")
# if nvidia_api_key is None:
#     raise ValueError("NVIDIA_API_KEY environment variable is not set.")
# os.environ['NVIDIA_API_KEY'] = nvidia_api_key

# # # Predefined set of questions
# # questions = [
# #     "1. Provide a detailed description of the proposed solution.",
# #     "2. What are the scalability options for the proposed solution?",
# #     "3. Scalability - min and max capacity and number of objects supported",
# #     "4. No SPOF (Single Point of Failure) components – for software defined, how does your software report and manage failed components? Are there any SPOF in any piece of the architecture - hardware, software, or both? If no, please explain how this is accomplished.",
# #     "5. Does the solution have automatic failure notification via email, text, and other methods?",
# #     "6. Can access to reporting features be segregated by users/roles? If so, please explain.",
# #     "7. Does the solution have reporting related to system health? If so, please explain the reporting and the granularity.",
# #     "8. Product reporting - including chargeback for tenant capacity, performance usage and product health.",
# #     "9. Describe all the security features of the proposed solution.",
# #     "10. Please specify how data at rest encryption is provided (e.g. underlying server/disk encryption, software-based server-side encryption, client provided Keys?)",
# #     "11. Does the solution support FIPS 140-2 compliant AES 256-bit encryption? Is this compliance selectable or does it demand all or nothing?",
# #     "12. Product performance capabilities - e.g. read or written per minute. How many connections can the product sustain?",
# #     "13. Availability and Reliability (Resilience)",
# #     "14. Load balancing customer IO capabilities across nodes in a cluster or across geographic areas within a region.",
# #     "15. Replication capabilities between datacentres / data protection against data loss and corruption.",
# #     "16. Compression and deduplication capabilities.",
# #     "17. TCP / Application Protocol Support.",
# #     "18. RESTful API integration for product management.",
# #     "19. Can reports be produced via REST API or viewed on a GUI - based on node/site and global capacity (total, quota, used, free)? If so, please explain what is available to see and provide sample reports.",
# #     "20. Administration capabilities for a multi-tenant product / solution.",
# #     "21. Product monitoring capabilities - device performance, user experience and storage capacity; over which protocols and interfaces.",
# #     "22. Ability to set and apply archive quotas per tenant.",
# #     "23. Data Tiering Capability, including cloud storage endpoints / Hybrid solution.",
# #     "24. Cloud Interoperability with mainstream Cloud hyperscaler.",
# #     "25. While there is no requirement for the solution to fit with a private cloud infrastructure, this is a chance for the proponent to showcase its product’s features. It will be used to ensure that the products fit well into the company's future direction."
# # ]

# EMBEDDINGS_FILE = 'NetApp.pkl'

# def vector_embedding():
#     if 'vectors' not in st.session_state:
#         if os.path.exists(EMBEDDINGS_FILE):
#             with open(EMBEDDINGS_FILE, 'rb') as f:
#                 st.session_state.vectors = pickle.load(f)
#             st.write("Loaded precomputed embeddings from file.")
#         else:
#             st.session_state.embeddings = NVIDIAEmbeddings()
#             st.session_state.loader = PyPDFDirectoryLoader("./us_census")  # Data Ingestion
#             st.session_state.docs = st.session_state.loader.load()  # Document Loading
#             st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
#             st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])  # Splitting
#             st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
            
#             # Save embeddings to file
#             with open(EMBEDDINGS_FILE, 'wb') as f:
#                 pickle.dump(st.session_state.vectors, f)
#             st.write("Computed and saved embeddings to file.") 
        
# st.set_page_config(page_title="Nvidia NIM Demo", layout="wide")
# st.title("Nvidia NIM Demo")
# st.markdown("This demo showcases document embedding and retrieval using NVIDIA AI.")

# llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# prompt = ChatPromptTemplate.from_template(
# """
# Answer the questions based on the provided context only and make sure to remember below points:
# >If there is no Answer available; the output should be “No Answer”
# >If Partial Answer is available; The output should include “Partial Answer Available”
# >Answer length should be not exceed more than 75-80 words or 7-8 lines.
# >Simple formatting can be applied for readability and evaluation
# Please provide the most accurate response based on the question.
# <context>
# {context}
# <context>
# Questions: {input}
# """
# )

# # Add file uploader for Excel file with questions
# uploaded_file = st.file_uploader("Upload an Excel file with questions", type=["xlsx"])

# if st.button("Documents Embedding"):
#     vector_embedding()
#     st.write("Vector Store DB Is Ready")

# if "vectors" in st.session_state:
#     if uploaded_file is not None:
#         # Read questions from the uploaded Excel file
#         questions_df = pd.read_excel(uploaded_file)
#         questions = questions_df['Sample RFP Questions asked by Customer'].tolist()  
#     else:
#         st.warning("Please upload an Excel file containing the questions.")
#         questions = []  # No questions to process

#     if questions:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)
#         responses = []

#         for question in questions:
#             response = retrieval_chain.invoke({'input': question})
#             answer = response.get('answer', 'No Answer')
#             responses.append({'Question': question, 'Answer': answer})

#         df = pd.DataFrame(responses)
#         st.write(df)
#     # with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#     #     df.to_excel(writer, index=False, sheet_name='Sheet1')
#     #     writer.save()
#     output = io.BytesIO()
#     with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#         df.to_excel(writer, index=False, sheet_name='Sheet1')


#     st.download_button(
#         label="Download Responses as Excel",
#         data=output.getvalue(),
#         file_name='responses.xlsx',
#         mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
#     )

#     with st.expander("Document Similarity Search"):
#         for doc in responses:
#             st.write(doc['Answer'])
#             st.write("--------------------------------")


##Integrated & Updated code:

import streamlit as st
import os
import pickle
import pandas as pd
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
if nvidia_api_key is None:
    raise ValueError("NVIDIA_API_KEY environment variable is not set.")

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only and make sure to remember below points:
>If there is no Answer available; the output should be “No Answer”
>If Partial Answer is available; The output should include “Partial Answer Available”
>Answer length should not exceed more than 75-80 words or 7-8 lines.
>Simple formatting can be applied for readability and evaluation
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

tabs = st.tabs(["Hitachi App", "NetApp"])

with tabs[0]:
    st.title("Nvidia NIM DEMO : Hitachi App")
    st.markdown("This demo showcases document embedding and retrieval using NVIDIA AI for Hitachi.")

    EMBEDDINGS_FILE_HITACHI = 'Hitachi_vectors.pkl'
    DATA_DIR_HITACHI = r"C:\Users\Mani_Kandan_Raja\Desktop\Nvidia_NIM\Nvidia-NIM\RFP_Files\HCP - Administer, Namespaces and Mgmt API Largem Management Help.pdf"

    uploaded_file = st.file_uploader("Upload an Excel file with questions", type=["xlsx"], key="hitachi")
    if st.button("Documents Embedding", key="hitachi_embed"):
        # vector_embedding(EMBEDDINGS_FILE_HITACHI, DATA_DIR_HITACHI)
        if 'Hitachi_vectors' in st.session_state:
            if os.path.exists(EMBEDDINGS_FILE_HITACHI):
                with open(EMBEDDINGS_FILE_HITACHI, 'rb') as f:
                    st.session_state.Hitachi_vectors = pickle.load(f)
                st.write("Loaded precomputed embeddings from file.")
                                    
        else:
                st.session_state.embeddings = NVIDIAEmbeddings() 
                st.session_state.loader = PyPDFLoader(DATA_DIR_HITACHI)  # Data Ingestion
                st.session_state.docs = st.session_state.loader.load()  # Document Loading
                st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
                st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
                st.session_state.Hitachi_vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
                
                # Save embeddings to file
                with open(EMBEDDINGS_FILE_HITACHI, 'wb') as f:
                    pickle.dump(st.session_state.Hitachi_vectors, f)
                st.write("Computed and saved embeddings to file.")
        st.write("Vector Store DB Is Ready : Hitachi_vectors.pkl")
        
        if uploaded_file is not None:
            questions_df = pd.read_excel(uploaded_file)
            questions = questions_df['Sample RFP Questions asked by Customer'].tolist()  
        else:
            st.warning("Please upload an Excel file containing the questions.")
            questions = []  # No questions to process

        if questions:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.Hitachi_vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            responses = []

            for question in questions:
                response = retrieval_chain.invoke({'input': question})
                answer = response.get('answer', 'No Answer')
                responses.append({'Question': question, 'Answer': answer})

            df = pd.DataFrame(responses)
            st.write(df)    

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')   

            st.download_button(
                label="Download Responses as Excel",
                data=output.getvalue(),
                file_name='Hitachi_responses.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            with st.expander("Document Similarity Search"):
                for doc in responses:
                    st.write(doc['Answer'])
                    st.write("--------------------------------")



with tabs[1]:
    st.title("Nvidia NIM DEMO : NetApp")
    st.markdown("This demo showcases document embedding and retrieval using NVIDIA AI for NetApp.")

    EMBEDDINGS_FILE_NETAPP = 'NetApp.pkl'
    DATA_DIR_NETAPP = r"C:\Users\Mani_Kandan_Raja\Desktop\Nvidia_NIM\Nvidia-NIM\RFP_Files\NetApp StorageGrid 11 8 Complete Document.pdf"

    uploaded_file = st.file_uploader("Upload an Excel file with questions", type=["xlsx"], key="netapp")
    if st.button("Documents Embedding", key="netapp_embed"):
        # vector_embedding(EMBEDDINGS_FILE_HITACHI, DATA_DIR_HITACHI)
        if 'NetApp' in st.session_state:
            if os.path.exists(EMBEDDINGS_FILE_NETAPP):
                with open(EMBEDDINGS_FILE_NETAPP, 'rb') as f:
                    st.session_state.NetApp = pickle.load(f)
                st.write("Loaded precomputed embeddings from file.")
                
        else:
            st.session_state.embeddings = NVIDIAEmbeddings()
            st.session_state.loader = PyPDFLoader(DATA_DIR_NETAPP)  # Data Ingestion
            st.session_state.docs = st.session_state.loader.load()  # Document Loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)  # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Splitting
            st.session_state.NetApp = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings
            
            # Save embeddings to file
            with open(EMBEDDINGS_FILE_NETAPP, 'wb') as f:
                pickle.dump(st.session_state.NetApp, f)
            st.write("Computed and saved embeddings to file.")
        st.write("Vector Store DB Is Ready : NetApp_vectors.pkl")
                
        if uploaded_file is not None:
            questions_df = pd.read_excel(uploaded_file)
            questions = questions_df['Sample RFP Questions asked by Customer'].tolist()  
        else:
            st.warning("Please upload an Excel file containing the questions.")
            questions = [] 

        if questions:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.NetApp.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            responses = []

            for question in questions:
                response = retrieval_chain.invoke({'input': question})
                answer = response.get('answer', 'No Answer')
                responses.append({'Question': question, 'Answer': answer})

            df = pd.DataFrame(responses)
            st.write(df)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Sheet1')

            st.download_button(
                label="Download Responses as Excel",
                data=output.getvalue(),
                file_name='NetApp_responses.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            with st.expander("Document Similarity Search"):
                for doc in responses:
                    st.write(doc['Answer'])
                    st.write("--------------------------------")
        

   