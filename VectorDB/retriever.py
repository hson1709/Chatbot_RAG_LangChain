from langchain_chroma import Chroma
from Model.model import embedding_model, persist_directory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from Model.model import compressor

db_thuoc = Chroma(
    collection_name="thuoc",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

db_benh = Chroma(
    collection_name="benh",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

def query_db(query, data_type="thuoc", top_k=2):

    chroma_db = Chroma(
        collection_name=data_type,
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    retriver = chroma_db.as_retriever(search_kwargs={"k": top_k})

    results = retriver.invoke(query)

    print(results)


def reranking_test(query, data_type):

    if data_type == "nhieu":
        retriever_thuoc = db_thuoc.as_retriever(search_type="mmr", search_kwargs={"k": 30})
        retriever_benh = db_benh.as_retriever(search_type="mmr", search_kwargs={"k": 30})
        ensemble_retriever = EnsembleRetriever(retrievers=[retriever_thuoc, retriever_benh])
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
    else:
        db = db_thuoc if data_type == "thuoc" else db_benh
        retriever = db.as_retriever(search_kwargs={"k": 30})
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )

    result = compression_retriever.invoke(query)
    print(result)
    
    return compression_retriever

#query_db("Beprosone Cream có tác dụng gì ? Tôi bị viêm da cơ địa, có dùng được không?", data_type="thuoc")
#query_db("Suy tim là bệnh gì ?", data_type="benh")
reranking_test("Tôi thấy cơ thể mệt mỏi, sụt cân, thường xuyên sốt kéo dài, đổ mồ hôi và trí nhớ trở nên kém đi. Đây là dấu hiệu của bệnh gì?", data_type="benh")
