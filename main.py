import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
import time

# ==============================================================================
# PHẦN 1: ĐỊNH NGHĨA CÁC LỚP (CLASSES)
# Sử dụng lại các class bạn đã cung cấp
# ==============================================================================

class Route:
    """
    Đại diện cho một route trong hệ thống, bao gồm tên và các câu mẫu.
    """
    def __init__(self, name: str = None, samples: List = []):
        self.name = name
        self.samples = samples

class SemanticRouter:
    """
    Thực hiện việc định tuyến ngữ nghĩa (semantic routing) dựa trên embedding.
    """
    def __init__(self, embedding_model, routes: List[Route]):
        self.routes = routes
        self.embedding_model = embedding_model
        self.routes_embedding = {}

        # Pre-compute embeddings cho tất cả các câu mẫu trong mỗi route
        for route in self.routes:
            self.routes_embedding[route.name] = self.embedding_model.encode(
                route.samples,
                convert_to_tensor=True
            ).cpu().numpy()

    def get_routes(self) -> List[Route]:
        """Trả về danh sách các routes."""
        return self.routes

    def guide(self, query: str, threshold: float = 0.5) -> (float, str):
        """
        Định tuyến một câu truy vấn (query) đến route phù hợp nhất.
        """
        query_embedding = self.embedding_model.encode([query], convert_to_tensor=True).cpu().numpy()
        # Chuẩn hóa vector truy vấn
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        scores = []
        # Tính toán độ tương đồng cosine giữa truy vấn và các câu mẫu của mỗi route
        for route in self.routes:
            # Chuẩn hóa các vector của route
            route_embeddings = self.routes_embedding[route.name]
            route_embeddings_norm = route_embeddings / np.linalg.norm(route_embeddings, axis=1, keepdims=True)
            
            # Tính độ tương đồng cosine và lấy giá trị trung bình
            similarity_scores = np.dot(route_embeddings_norm, query_embedding.T).flatten()
            mean_score = np.mean(similarity_scores)
            scores.append((mean_score, route.name))

        # Sắp xếp các route theo điểm số từ cao đến thấp
        scores.sort(reverse=True)
        
        # Lấy route có điểm cao nhất
        best_score, best_route_name = scores[0]
        
        return best_score, best_route_name

# ==============================================================================
# PHẦN 2: CHUẨN BỊ DỮ LIỆU GIẢ (MOCK DATA)
# ==============================================================================

# --- Route: products (Thông tin sản phẩm) ---
# Dữ liệu bạn đã cung cấp
productsSample = [
    "Bạn có sẵn iPhone mới nhất không?", "Giá của Samsung Galaxy S21 là bao nhiêu?",
    "OnePlus 9 Pro có màu xanh không?", "Thông số kỹ thuật của Google Pixel 6 là gì?",
    "Huawei P50 Pro có sẵn ở cửa hàng của bạn không?", "iPhone 13 có những màu gì?",
    "Có giảm giá nào cho Samsung Galaxy Note 20 không?", "Sony Xperia 1 III có sẵn không?",
    "Cửa hàng của bạn có Google Pixel mới nhất không?", "Có chương trình khuyến mãi nào cho OnePlus Nord không?",
    "Dung lượng lưu trữ của Samsung Galaxy S21 là bao nhiêu?", "iPhone SE 2022 có sẵn không?",
    "Sự khác biệt về giá giữa iPhone 12 và 13 là gì?", "Motorola Edge 20 có sẵn không?",
    "Google Pixel 6 Pro có sẵn ở cửa hàng của bạn không?", "Bạn có bán Xiaomi Mi 11 không?",
    "Có chương trình ưu đãi nào cho Samsung Galaxy Z Fold 3 không?", "Oppo Find X3 Pro có sẵn không?",
    "Các tính năng của iPhone 13 Pro là gì?", "LG Wing có sẵn không?", "Nokia 8.3 5G có sẵn không?",
    "Asus ROG Phone 5 có sẵn không?", "Bạn có Realme GT không?", "Vivo X60 Pro có sẵn không?",
    "Bạn có Honor 50 không?", "Tuổi thọ pin của Samsung Galaxy A52 là bao nhiêu?",
    "ZTE Axon 30 có sẵn không?", "Bạn có BlackBerry KEY2 không?",
    "Kích thước màn hình của iPhone 13 Mini là bao nhiêu?", "Bạn có TCL 20 Pro 5G không?",
    "Nokia XR20 có sẵn ở cửa hàng của bạn không?", "Giá của Samsung Galaxy S20 FE là bao nhiêu?",
    "Bạn có iPhone 12 Pro Max không?", "Redmi Note 10 Pro có sẵn không?",
    "Sự khác biệt giữa iPhone 12 và iPhone 13 là gì?", "Bạn có Sony Xperia 5 II không?",
    "Thời gian bảo hành của Samsung Galaxy S21 là bao lâu?", "Google Pixel 5a có sẵn không?",
    "Bạn có OnePlus 8T không?", "Giá của iPhone 13 Pro Max là bao nhiêu?",
    "Samsung Galaxy Z Flip 3 có sẵn không?", "Thông số kỹ thuật của Oppo Reno6 Pro là gì?",
    "Bạn có Vivo V21 không?", "Motorola Moto G100 có sẵn không?", "Bạn có Huawei Mate 40 Pro không?",
    "Realme 8 Pro có sẵn ở cửa hàng của bạn không?", "Asus Zenfone 8 có sẵn không?",
    "LG Velvet có sẵn không?", "Dung lượng lưu trữ của iPhone 12 là bao nhiêu?",
    "Bạn có Honor Magic 3 không?", "Xiaomi Mi 11 Ultra có sẵn không?"
]

# --- Route: chitchat (Trò chuyện/Hỏi đáp thông thường) ---
# Dữ liệu bạn đã cung cấp
chitchatSample = [
    "Thời tiết hôm nay như thế nào?", "Ngoài trời nóng bao nhiêu?", "Ngày mai có mưa không?",
    "Nhiệt độ hiện tại là bao nhiêu?", "Bạn có thể cho tôi biết điều kiện thời tiết hiện tại không?",
    "Cuối tuần này có nắng không?", "Nhiệt độ hôm qua là bao nhiêu?", "Đêm nay trời sẽ lạnh đến mức nào?",
    "Ai là tổng thống đầu tiên của Hoa Kỳ?", "Chiến tranh thế giới thứ hai kết thúc vào năm nào?",
    "Bạn có thể kể cho tôi về lịch sử của internet không?", "Tháp Eiffel được xây dựng vào năm nào?",
    "Ai đã phát minh ra điện thoại?", "Tên của bạn là gì?", "Bạn có tên không?",
    "Tôi nên gọi bạn là gì?", "Ai đã tạo ra bạn?", "Bạn bao nhiêu tuổi?",
    "Bạn có thể kể cho tôi một sự thật thú vị không?", "Bạn có biết bất kỳ câu đố thú vị nào không?",
    "Màu sắc yêu thích của bạn là gì?", "Bộ phim yêu thích của bạn là gì?",
    "Bạn có sở thích nào không?", "Ý nghĩa của cuộc sống là gì?", "Bạn có thể kể cho tôi một câu chuyện cười không?",
    "Thủ đô của Pháp là gì?", "Dân số thế giới là bao nhiêu?", "Có bao nhiêu châu lục?",
    "Ai đã viết 'Giết con chim nhại'?", "Bạn có thể cho tôi một câu nói của Albert Einstein không?", "bạn khỏe không?", "hôm nay là một ngày đẹp trời"
]

# --- Route: order_info (Thông tin đơn hàng) ---
# Dữ liệu giả tôi đã tạo thêm
orderInfoSample = [
    "Kiểm tra tình trạng đơn hàng của tôi.", "Đơn hàng của tôi đang ở đâu?",
    "Khi nào tôi sẽ nhận được hàng?", "Mã đơn hàng của tôi là ABC-12345.",
    "Tôi muốn xem lại chi tiết đơn hàng.", "Làm thế nào để theo dõi đơn hàng?",
    "Thông tin vận chuyển của đơn hàng là gì?", "Đơn hàng của tôi đã được gửi đi chưa?",
    "Tại sao đơn hàng của tôi bị trễ?", "Tôi có thể thay đổi địa chỉ giao hàng không?",
    "Cho tôi biết mã vận đơn.", "Đơn hàng #XYZ987 đang ở giai đoạn nào?",
    "Tôi muốn hủy đơn hàng này.", "Ai là người giao hàng cho tôi?",
    "Gói hàng của tôi có những gì?", "Lịch sử mua hàng của tôi.",
    "Tình trạng đơn hàng gần đây nhất của tôi là gì?", "Đơn hàng đã thanh toán chưa?",
    "Tôi có nhận được email xác nhận đơn hàng không?", "Bao lâu nữa thì hàng tới nơi?",
    "Xem giúp tôi đơn hàng đặt tuần trước.", "Tôi có thể xem lại các đơn hàng cũ không?",
    "Đơn hàng của tôi có vấn đề gì không?", "Phí vận chuyển cho đơn hàng này là bao nhiêu?",
    "Khi nào shipper sẽ gọi cho tôi?", "Tôi có thể hẹn lại ngày giao hàng không?",
    "Liệu đơn hàng có đến kịp cuối tuần không?", "Tôi chưa nhận được hàng dù đã báo giao thành công.",
    "Kiểm tra giúp tôi mã đơn 11223344.", "Đơn hàng của tôi do ai xử lý?"
]


# ==============================================================================
# PHẦN 3: XÂY DỰNG ỨNG DỤNG STREAMLIT
# ==============================================================================

st.set_page_config(page_title="Demo Semantic Router", layout="wide")

# Hàm để tải model và router, sử dụng cache của Streamlit để không phải tải lại mỗi lần
@st.cache_resource
def load_resources():
    """Tải model embedding và khởi tạo router."""
    with st.spinner("Đang tải model và khởi tạo router... Vui lòng chờ một lát."):
        # Sử dụng một model đa ngôn ngữ phù hợp cho tiếng Việt
        model = SentenceTransformer('intfloat/multilingual-e5-base')
        
        # Khởi tạo các routes
        routes = [
            Route(name="products", samples=productsSample),
            Route(name="order_info", samples=orderInfoSample),
            Route(name="chitchat", samples=chitchatSample)
        ]
        
        # Khởi tạo router
        router = SemanticRouter(embedding_model=model, routes=routes)
    return router

# Tải router
router = load_resources()

# --- Giao diện người dùng ---
st.title("🚀 Demo Semantic Router cho Chatbot CSKH")
st.markdown("""
Ứng dụng này mô phỏng cách một chatbot có thể sử dụng **semantic routing** để hiểu ý định của người dùng 
và điều hướng câu hỏi đến luồng xử lý phù hợp.
- **`products`**: Các câu hỏi liên quan đến thông tin, giá cả, tính năng của sản phẩm.
- **`order_info`**: Các câu hỏi về tình trạng đơn hàng, giao hàng, lịch sử mua hàng.
- **`chitchat`**: Các câu hỏi thông thường, ngoài lề, không thuộc hai loại trên.
""")
st.info("Mẹo: Hãy thử các câu hỏi khác nhau như 'Giá iPhone 13?', 'Đơn hàng của tôi đâu rồi?', 'Hôm nay bạn thế nào?' để xem kết quả.")


# Form để người dùng nhập câu hỏi
with st.form("query_form"):
    user_query = st.text_input(
        "Nhập câu hỏi của bạn vào đây:", 
        "",
        placeholder="Ví dụ: Kiểm tra giúp tôi đơn hàng #12345"
    )
    submitted = st.form_submit_button("Xác định Route")

if submitted and user_query:
    with st.spinner("Đang phân tích câu hỏi của bạn..."):
        time.sleep(1) # Giả lập thời gian xử lý
        
        # Sử dụng router để dự đoán
        score, route_name = router.guide(user_query)

        st.subheader("Kết quả phân tích:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Route được đề xuất", value=route_name.replace("_", " ").title())
        with col2:
            st.metric(label="Điểm tương đồng (Cosine Similarity)", value=f"{score:.4f}")

        # Hiển thị giải thích về route được chọn
        if route_name == "products":
            st.success("✅ **Luồng xử lý:** Câu hỏi này sẽ được chuyển đến hệ thống **Knowledge Base** để truy vấn thông tin về sản phẩm.")
        elif route_name == "order_info":
            st.info("✅ **Luồng xử lý:** Câu hỏi này sẽ được chuyển đến **máy chủ MCP** để truy vấn thông tin chi tiết về đơn hàng.")
        elif route_name == "chitchat":
            st.warning("✅ **Luồng xử lý:** Câu hỏi này sẽ được **LLM (Large Language Model)** trả lời trực tiếp như một cuộc trò chuyện thông thường.")
            
elif submitted and not user_query:
    st.error("Vui lòng nhập một câu hỏi để thực hiện định tuyến.")
