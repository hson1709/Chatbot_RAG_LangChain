from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableSequence
from Model.model import compressor

# Cấu hình prompt
def create_prompt_template():
    classification_prompt_template = """Bạn là một trợ lý y tế thông minh.
    Hãy phân loại truy vấn sau đây vào một trong bốn nhóm (chỉ trả lời "thuoc", "benh", "nhieu" hoặc "unknown"):
    1. "thuoc" nếu nó liên quan đến thông tin thuốc (thành phần, công dụng, liều lượng, tác dụng phụ, bảo quản, cách dùng).
    2. "benh" nếu nó liên quan đến bệnh lý (triệu chứng, nguyên nhân, chẩn đoán, điều trị, phòng ngừa).
    3. "nhieu" nếu nó có thể liên quan đến cả hai nhóm.
    4. "unknown" nếu không có thông tin nào liên quan đến thuốc hoặc bệnh.

    Truy vấn: {query}
    Phân loại:"""


    answer_prompt_template = """Bạn là BotTech — một trợ lý y tế thông minh, thân thiện và đáng tin cậy, chuyên hỗ trợ giải đáp các thắc mắc về bệnh lý và thuốc.
    Dựa vào thông tin được cung cấp bên dưới phần "Nội dung", hãy trả lời câu hỏi một cách chính xác, dễ hiểu và phù hợp với người dùng phổ thông.
    Hãy trả lời một cách thân thiện với người dùng.

    Nguyên tắc trả lời:
    - Chỉ sử dụng thông tin từ phần "Nội dung". Không tự tạo hoặc phỏng đoán thông tin.
    - Hãy cố gắng trả lời dựa vào thông tin sẵn có. Trong trường hợp phần "Nội dung" không chứa các thông tin về câu hỏi, hãy phản hồi:  
    "Xin lỗi, tôi chưa thể giải đáp thắc mắc của bạn. Vui lòng đặt câu hỏi cụ thể hoặc chi tiết hơn."
    - Nếu câu hỏi không liên quan đến lĩnh vực y tế, hãy phản hồi:  
    "Xin lỗi, tôi chỉ có thể cung cấp thông tin trong lĩnh vực y tế."
    - Nếu người dùng hỏi về bạn, hãy giới thiệu:  
    "Tôi là BotTech — một trợ lý y tế thông minh được phát triển để hỗ trợ người dùng về y tế."

    Nội dung:
    {context}

    Câu hỏi: {question}
    Trả lời:"""

    ANSWER_PROMPT = PromptTemplate(template=answer_prompt_template, input_variables=["context", "question"])
    CLASSIFICATION_PROMPT = PromptTemplate(template=classification_prompt_template, input_variables=["query"])

    return ANSWER_PROMPT, CLASSIFICATION_PROMPT

def create_conversational_chain(llm, retriever, memory, answer_prompt):
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": answer_prompt}
    )
    return chain



# Hàm xác định loại truy vấn bằng Gemini
def detect_data_type(question, classification_prompt, llm):
    classification_chain = RunnableSequence(classification_prompt | llm)  
    data_type = classification_chain.invoke({"query": question}) 
    return data_type


# Reranking các retriver
def reranking(db_thuoc, db_benh, data_type):

    if data_type == "nhieu":
        retriever_thuoc = db_thuoc.as_retriever(search_type="mmr", search_kwargs={"k": 20})
        retriever_benh = db_benh.as_retriever(search_type="mmr", search_kwargs={"k": 20})
        ensemble_retriever = EnsembleRetriever(retrievers=[retriever_thuoc, retriever_benh])
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
    else:
        db = db_thuoc if data_type == "thuoc" else db_benh
        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 20})
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
    return compression_retriever

