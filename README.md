# RAG y tế với LangChain

![image](https://github.com/hson1709/Chatbot_RAG_LangChain/blob/master/UI.png)

## Giới Thiệu

Dự án này là một RAG (Retrieval-Augmented Generation) chuyên về lĩnh vực y tế, sử dụng **LangChain**, **Flask** và **Google Gemini API** để giải đáp thông tin về các loại bệnh và thuốc.

Chatbot có khả năng:
- Tìm kiếm và trích xuất thông tin thuốc và bệnh từ kho dữ liệu.
- Sử dụng **ChromaDB** để lưu trữ vector embeddings.
- Dùng Reranking với **Cross-Encoder Reranker** và để tối ưu kết quả tìm kiếm.
- Tự động phân loại truy vấn và chọn nguồn dữ liệu phù hợp.
- Sử dụng **Google Gemini** để sinh ra câu trả lời có ngữ cảnh.
- Giao tiếp qua giao diện web trực quan

## Mô hình
- Embedding Model: all-mpnet-base-v2
- Reranking Model: ms-marco-MiniLM-L-6-v2
- LLM: gemini-2.0-pro-exp

## Dữ liệu

Dữ liệu chatbot được thu thập từ:
- **Thuốc:** Cào từ trang [Nhà thuốc Long Châu](https://nhathuoclongchau.com.vn)
- **Bệnh:** Cào từ trang [Vinmec](https://www.vinmec.com/vie/tra-cuu-benh/)

### Các trường dữ liệu
- **Dữ liệu thuốc:** Tên thuốc, URL, Thành phần, Công dụng, Cách dùng, Tác dụng phụ, Lưu ý, Bảo quản, Loại thuốc.
- **Dữ liệu bệnh:** Nguyên nhân, Triệu chứng, Đối tượng nguy cơ, Phòng ngừa, Chẩn đoán, Điều trị.

## Cài Đặt & Chạy Dự Án

### 1. Cài Đặt Môi Trường
Yêu cầu Python 3.8+

```bash
pip install -r requirements.txt
```

### 2. Thêm Google API Key
Tạo file .env như sau:

```bash
GOOGLE_API_KEY = "YOUR API KEY"  # Thay bằng API key của bạn
```

### 3. Tạo Vector Database
Chạy các lệnh sau để tạo vector database từ dữ liệu thuốc và bệnh:

```python
python create_db.py
```

### 4. Chạy Chatbot

```python
python app.py
```
