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
# --- Route: products (Thông tin sản phẩm - Tổng quát) ---
# Dữ liệu tập trung vào các loại câu hỏi về sản phẩm, không phụ thuộc vào ngành hàng.
productsSample = [
    "Sản phẩm này còn hàng không?",
    "Cho tôi biết giá của món đồ này.",
    "Sản phẩm này có những màu nào khác?",
    "Thông tin chi tiết về sản phẩm này là gì?",
    "Bên bạn có bán mặt hàng [tên một mặt hàng] không?",
    "Chất liệu của sản phẩm này là gì?",
    "Tư vấn cho tôi một sản phẩm tương tự.",
    "Chính sách bảo hành cho mặt hàng này như thế nào?",
    "Sản phẩm này có được giảm giá không?",
    "Tôi có thể xem ảnh thật của sản phẩm không?",
    "Thời gian bảo hành là bao lâu?",
    "Hướng dẫn sử dụng sản phẩm này.",
    "So sánh giúp tôi sản phẩm này với sản phẩm kia.",
    "Sản phẩm nào đang bán chạy nhất ở cửa hàng?",
    "Sản phẩm này sản xuất ở đâu?",
    "Thành phần của sản phẩm gồm những gì?",
    "Tôi có thể đổi trả hàng nếu không ưng ý không?",
    "Kích thước của sản phẩm này là bao nhiêu?",
    "Khi nào thì có hàng lại?",
    "Phí vận chuyển cho sản phẩm này đến Hà Nội là bao nhiêu?",
    "Bạn có thể giới thiệu một món quà cho nam không?",
    "Mã của sản phẩm này là gì?",
    "Sản phẩm này có an toàn cho trẻ em không?",
    "Hạn sử dụng của mặt hàng này đến khi nào?",
    "Có những phụ kiện nào đi kèm sản phẩm không?"
]

# --- Route: order_info (Thông tin đơn hàng) ---
# Dữ liệu này vốn đã khá tổng quát và giữ nguyên để đảm bảo tính nhất quán.
orderInfoSample = [
    "Kiểm tra tình trạng đơn hàng của tôi.",
    "Đơn hàng của tôi đang ở đâu?",
    "Khi nào tôi sẽ nhận được hàng?",
    "Mã đơn hàng của tôi là ABC-12345.",
    "Tôi muốn xem lại chi tiết đơn hàng.",
    "Làm thế nào để theo dõi đơn hàng?",
    "Thông tin vận chuyển của đơn hàng là gì?",
    "Đơn hàng của tôi đã được gửi đi chưa?",
    "Tại sao đơn hàng của tôi bị trễ?",
    "Tôi có thể thay đổi địa chỉ giao hàng không?",
    "Cho tôi biết mã vận đơn.",
    "Đơn hàng #XYZ987 đang ở giai đoạn nào?",
    "Tôi muốn hủy đơn hàng này.",
    "Ai là người giao hàng cho tôi?",
    "Gói hàng của tôi có những gì?",
    "Lịch sử mua hàng của tôi.",
    "Tình trạng đơn hàng gần đây nhất của tôi là gì?",
    "Đơn hàng đã thanh toán chưa?",
    "Tôi có nhận được email xác nhận đơn hàng không?",
    "Bao lâu nữa thì hàng tới nơi?",
    "Xem giúp tôi đơn hàng đặt tuần trước.",
    "Tôi có thể xem lại các đơn hàng cũ không?",
    "Đơn hàng của tôi có vấn đề gì không?",
    "Phí vận chuyển cho đơn hàng này là bao nhiêu?",
    "Khi nào shipper sẽ gọi cho tôi?"
]

# --- Route: chitchat (Trò chuyện/Hỏi đáp thông thường) ---
# Giữ nguyên dữ liệu chitchat.
chitchatSample = [
    "Thời tiết hôm nay như thế nào?",
    "Kể cho tôi một câu chuyện cười đi.",
    "Bạn là ai?",
    "Bạn có thể làm gì?",
    "Thủ đô của Việt Nam là gì?",
    "Hôm nay là một ngày đẹp trời.",
    "Cảm ơn bạn nhé.",
    "Ý nghĩa của cuộc sống là gì?",
    "Bạn có sở thích nào không?",
    "Ai đã tạo ra bạn?",
    "Trời hôm nay nóng quá.",
    "Bạn có biết nấu ăn không?",
    "Chúc một ngày tốt lành.",
    "Tôi nên gọi bạn là gì?",
    "Bạn bao nhiêu tuổi?"
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
        
        # <<< THAY ĐỔI 2: Bắt đầu tính giờ >>>
        start_time = time.monotonic()
        
        # Sử dụng router để dự đoán
        score, route_name = router.guide(user_query)

        # <<< THAY ĐỔI 2: Kết thúc tính giờ và tính toán thời gian xử lý >>>
        end_time = time.monotonic()
        processing_time = end_time - start_time

        st.subheader("Kết quả phân tích:")
        
        # <<< THAY ĐỔI 3: Chia layout thành 3 cột và thêm metric thời gian >>>
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Route được đề xuất", value=route_name.replace("_", " ").title())
        with col2:
            st.metric(label="Điểm tương đồng", value=f"{score:.4f}")
        with col3:
            st.metric(label="Thời gian xử lý", value=f"{processing_time:.4f} giây")

        # Hiển thị giải thích về route được chọn
        if route_name == "products":
            st.success("✅ **Luồng xử lý:** Câu hỏi này sẽ được chuyển đến hệ thống **Knowledge Base** để truy vấn thông tin về sản phẩm.")
        elif route_name == "order_info":
            st.info("✅ **Luồng xử lý:** Câu hỏi này sẽ được chuyển đến **máy chủ MCP** để truy vấn thông tin chi tiết về đơn hàng.")
        elif route_name == "chitchat":
            st.warning("✅ **Luồng xử lý:** Câu hỏi này sẽ được **LLM (Large Language Model)** trả lời trực tiếp như một cuộc trò chuyện thông thường.")
            
elif submitted and not user_query:
    st.error("Vui lòng nhập một câu hỏi để thực hiện định tuyến.")


