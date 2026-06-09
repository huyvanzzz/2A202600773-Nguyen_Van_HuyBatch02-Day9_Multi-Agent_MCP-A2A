# Bản Tóm Tắt Chi Tiết Yêu Cầu Thực Hành (Codelab Multi-Agent)

Bản tài liệu này tóm tắt toàn bộ các yêu cầu, bài tập và thử thách cần hoàn thành được trích xuất từ các tài liệu hướng dẫn của dự án (`README.md`, `CODELAB.md`, `INSTRUCTOR_GUIDE.md`, `exercises/README.md`).

---

## 🛠️ Yêu Cầu Chuẩn Bị & Thiết Lập Hệ Thống

Trước khi bắt đầu thực hiện các bài tập, cần đảm bảo môi trường lập trình được thiết lập đầy đủ:

1. **Phiên bản Python**: Yêu cầu cài đặt **Python 3.11 trở lên**.
2. **Trình quản lý thư viện**: Sử dụng [uv](https://docs.astral.sh/uv/) làm package manager.
3. **Cài đặt thư viện**: Chạy lệnh `uv sync` tại thư mục gốc của dự án.
4. **Cấu hình biến môi trường**:
   - Sao chép file mẫu: `cp .env.example .env`
   - Cập nhật key `OPENROUTER_API_KEY` (lấy từ [OpenRouter](https://openrouter.ai)).
   - Kiểm tra kết nối LLM qua lệnh: `uv run python stages/stage_1_direct_llm/main.py`.

---

## 📝 Chi Tiết Các Bài Tập Bắt Buộc (Exercises)

Hệ thống bài tập thực hành tập trung vào việc phát triển kỹ năng từ cấp độ đơn lẻ (Single Agent với Tools) lên hệ thống phân tán phức tạp (Multi-Agent).

### 1. Exercise 2: Thêm Tools và Knowledge Base (10 phút)
* **File nguồn cần chỉnh sửa**: [exercise_2_tools.py](file:///d:/Vin/2A202600773-Nguyen_Van_HuyBatch02-Day9_Multi-Agent_MCP-A2A/exercises/exercise_2_tools.py)
* **Lệnh chạy thử**: `uv run python exercises/exercise_2_tools.py`

#### Danh sách nhiệm vụ chi tiết:
* **Nhiệm vụ 2.1**: Thêm một bản ghi tri thức (entry) mới về luật lao động Việt Nam vào danh sách `LEGAL_KNOWLEDGE` (từ dòng 19):
  - **Mã định danh (id)**: `"labor_law"`
  - **Từ khóa gợi nhớ (keywords)**: `["lao động", "sa thái", "hợp đồng lao động", "labor", "termination"]`
  - **Nội dung tri thức (text)**: Quy định về việc người sử dụng lao động đơn phương chấm dứt hợp đồng lao động theo Bộ luật Lao động Việt Nam 2019 (ví dụ: do ốm đau dài ngày, thường xuyên không hoàn thành công việc, thiên tai, tuổi nghỉ hưu...).
* **Nhiệm vụ 2.2**: Định nghĩa một tool mới tên `check_statute_of_limitations` bằng cách sử dụng decorator `@tool`:
  - **Input**: `case_type: str` (loại vụ án: `contract`, `tort`, `property`).
  - **Logic**: Trả về thời hiệu khởi kiện tương ứng:
    - `"contract"` -> `"4 năm (UCC § 2-725)"`
    - `"tort"` -> `"2-3 năm tùy bang"`
    - `"property"` -> `"5 năm"`
    - Các trường hợp khác -> `"Không xác định"`
* **Nhiệm vụ 2.3**: Tích hợp tool mới vào quy trình gọi LLM:
  - Thêm tool `check_statute_of_limitations` vào mảng `tools` để bind với LLM thông qua `.bind_tools()`.
  - Cập nhật cấu trúc điều kiện trong hàm `main()` để xử lý việc kích hoạt tool này khi LLM yêu cầu (nếu `tool_call["name"] == "check_statute_of_limitations"`).

---

### 2. Exercise 4: Thêm Privacy Agent vào Multi-Agent System (15 phút)
* **File nguồn cần chỉnh sửa**: [exercise_4_multiagent.py](file:///d:/Vin/2A202600773-Nguyen_Van_HuyBatch02-Day9_Multi-Agent_MCP-A2A/exercises/exercise_4_multiagent.py)
* **Lệnh chạy thử**: `uv run python exercises/exercise_4_multiagent.py`

#### Danh sách nhiệm vụ chi tiết:
* **Nhiệm vụ 4.1**: Cập nhật cấu trúc trạng thái chung (`class State(TypedDict)`):
  - Thêm thuộc tính `privacy_analysis: Annotated[str, _last_wins]` để lưu trữ phân tích bảo mật dữ liệu.
* **Nhiệm vụ 4.2**: Xây dựng hàm `privacy_agent(state: State) -> dict`:
  - Đóng vai trò là chuyên gia bảo mật dữ liệu và luật GDPR.
  - Sử dụng LLM (`get_llm()`) để trả lời dựa trên câu hỏi gốc (`state['question']`) kết hợp với phân tích pháp lý tổng quát của `law_agent` (`state.get('law_analysis')`).
  - Trả về kết quả dưới dạng dict: `{"privacy_analysis": response.content}`.
* **Nhiệm vụ 4.3**: Bổ sung điều kiện định tuyến (conditional routing) trong hàm `check_routing(state: State)`:
  - Kiểm tra xem câu hỏi có chứa các từ khóa liên quan đến bảo mật thông tin hay không: `["data", "privacy", "gdpr", "dữ liệu"]`.
  - Nếu có, đẩy thêm task `Send("privacy_agent", state)` vào hàng đợi xử lý song song.
* **Nhiệm vụ 4.4**: Cập nhật hàm tổng hợp kết quả `aggregate_results(state: State)`:
  - Kiểm tra sự tồn tại của `privacy_analysis` trong trạng thái.
  - Nếu có, định dạng và bổ sung thông tin này vào báo cáo tổng hợp: `🔒 PHÂN TÍCH BẢO MẬT/GDPR:\n{state['privacy_analysis']}`.
* **Nhiệm vụ 4.5**: Cấu hình lại đồ thị LangGraph trong `build_graph()`:
  - Đăng ký node mới: `graph.add_node("privacy_agent", privacy_agent)`.
  - Thiết lập liên kết đồ thị: Thêm luồng kết nối cạnh (edge) từ `"privacy_agent"` đến node tổng hợp `"aggregate_results"`.

---

## 🚀 Thử Thách Nâng Cao (Optional Challenges)

Dành cho học viên hoàn thành sớm phần bài tập bắt buộc để mở rộng năng lực hệ thống:

| Thử Thách | Mô Tả Yêu Cầu |
| :--- | :--- |
| **Challenge 1: Financial Agent** | Tạo và tích hợp một agent chuyên trách phân tích tổn thất tài chính/định lượng thiệt hại kinh tế vào đồ thị. |
| **Challenge 2: Conversation Memory** | Tích hợp cơ chế lưu trữ lịch sử hội thoại (Session Memory) để agent có thể hiểu ngữ cảnh của các câu hỏi tiếp theo. |
| **Challenge 3: Custom Tool API** | Nâng cấp tool tra cứu bằng cách kết nối trực tiếp với các API cơ sở dữ liệu pháp lý trực tuyến thực tế. |
| **Challenge 4: Error Handling & Retry** | Xây dựng cơ chế bắt lỗi ngoại lệ, tự động thử lại (retry) với exponential backoff khi các cuộc gọi API ngoài thất bại. |
| **Challenge 5: Monitoring & Observability** | Tích hợp các công cụ theo dõi giám sát như LangSmith hoặc Prometheus để đo lường hiệu suất và luồng đi của agent. |

---

## 🔎 Câu Hỏi Ôn Tập Cần Trả Lời (Q&A)

Cuối buổi lab, học viên cần hiểu rõ và trả lời được 4 câu hỏi lý thuyết cốt lõi sau:
1. Khi nào nên áp dụng mô hình **Single Agent** và khi nào nên chuyển sang **Multi-Agent**?
2. Sự khác biệt và ưu thế của **A2A Protocol** so với các kiến trúc giao tiếp truyền thống như REST API thông thường hoặc gRPC?
3. Làm thế nào để ngăn chặn các vòng lặp ủy thác vô hạn (**infinite delegation loops**) khi các Agent tự động giao tiếp với nhau qua A2A?
4. Tại sao cần có một **Registry Service**? Chúng ta có thể dùng phương pháp cấu hình cứng địa chỉ URL (hardcode URLs) của các Agent được không?


