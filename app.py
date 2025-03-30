from Chatbot.chain import detect_data_type, create_prompt_template,reranking
from Chatbot.conversation import create_conversational_chain, create_response, memory
from VectorDB.load_db import load_db
from Model.model import llm

### Kiểm tra thử nghiệm

# Load vector database
db_thuoc = load_db("thuoc")
db_benh = load_db("benh")

# Câu hỏi
question = "Bạn là ai?"

# Tạo prompt
answer_prompt, classification_prompt = create_prompt_template()

# Phân loại data
data_type = detect_data_type(question, classification_prompt, llm)

# Tạo retriever
retriever = reranking(db_thuoc, db_benh, data_type)

# Tạo chain với bộ nhớ
chain = create_conversational_chain(llm, retriever, memory, answer_prompt)

# Tạo response
response = create_response(question, chain)

print("Trả lời:", response)
print("Phân loại:", data_type)
