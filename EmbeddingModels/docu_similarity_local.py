from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings":True}
)

documents = [
    "A savings account lets you keep your money safe in a bank while earning a small interest, usually around 2â€“4% per year.",
    "A current account is mainly for businesses. It allows unlimited daily transactions but doesn't pay interest on the balance.",
    "A fixed deposit lets you lock in a sum of money for a set period to earn a higher interest rate than a regular savings account.",
    "A cheque is a written instruction to your bank to pay a specific amount to someone else from your account.",
    "A joint account is shared between two or more people. All account holders can access and manage the funds together.",
    "Your account number is a unique code your bank gives your account. It's used to identify you during any transaction.",
    "A cheque book is a set of blank cheques given by your bank. You fill and sign them to make payments to others."
]

query = "What is current account?"

doc_embeddings =  embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

score = cosine_similarity([query_embedding], doc_embeddings)[0]

index , score = (sorted(list(enumerate(score)), key=lambda x:x[1])[-1])
print(query)
print(documents[index])
print("similarity score is:", score)