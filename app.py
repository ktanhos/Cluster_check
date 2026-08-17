import os

import pandas as pd
import streamlit as st

APP_VERSION = "USER-FRIENDLY-2026-08-17-01"

from backtest import calculate_forward_returns, summarize_forward_returns
from charts import behavior_map, cluster_count_chart, migration_heatmap
from clustering import rolling_cluster
from config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_CONFIRMATION_STEPS, DEFAULT_END, DEFAULT_K, DEFAULT_START, DEFAULT_STEP, DEFAULT_TRAIN_WINDOW
from data_loader import load_market_data
from features import build_feature_panel
from membership import change_table, symbols_for_period
from migration import build_migration_table

st.set_page_config(page_title="VN30 Hành vi thị trường", layout="wide")
st.title("VN30 Hành vi thị trường")
st.caption("Ứng dụng nhóm các cổ phiếu VN30 theo cách chúng đang vận động trên thị trường, chỉ dựa trên giá và khối lượng. Đây là công cụ nghiên cứu, không phải công cụ khuyến nghị mua bán.")

with st.sidebar:
    st.header("1. Dữ liệu")
    api_key = st.text_input(
        "Mã truy cập VNstock",
        value=st.session_state.get("vnstock_api_key", ""),
        type="password",
        placeholder="Dán mã truy cập VNstock",
        help="Mã truy cập chỉ được giữ trong phiên làm việc hiện tại.",
    )
    if api_key:
        st.session_state["vnstock_api_key"] = api_key

    start = st.date_input("Từ ngày", DEFAULT_START)
    end = st.date_input("Đến ngày", DEFAULT_END)
    update_data = st.button("Tải dữ liệu mới", type="primary", use_container_width=True)
    st.caption("Chỉ khi bấm nút này ứng dụng mới gọi VNstock. Sau khi tải xong, thay đổi thiết lập bên dưới không gọi lại dữ liệu.")

    st.divider()
    st.header("2. Cách nhận diện nhóm")
    train_window = st.number_input(
        "Số phiên dùng để nhận diện hành vi",
        min_value=60,
        max_value=504,
        value=DEFAULT_TRAIN_WINDOW,
        step=5,
        help="Số phiên giao dịch gần nhất được dùng để nhận diện trạng thái của thị trường. Ví dụ 60 tương đương khoảng 3 tháng giao dịch.",
    )
    k = st.number_input(
        "Số nhóm hành vi",
        min_value=2,
        max_value=8,
        value=DEFAULT_K,
        step=1,
        help="Ứng dụng sẽ chia các cổ phiếu thành số nhóm này. Mặc định là 4 nhóm theo thử nghiệm hiện tại.",
    )
    step = st.number_input(
        "Cứ bao nhiêu phiên cập nhật một lần",
        min_value=1,
        max_value=20,
        value=DEFAULT_STEP,
        step=1,
        help="Ví dụ 5 nghĩa là cứ 5 phiên giao dịch ứng dụng tạo một lần quan sát mới.",
    )

    with st.expander("Thiết lập nâng cao", expanded=False):
        confirmation_steps = st.number_input(
            "Một lần chuyển nhóm cần được giữ trong bao nhiêu lần quan sát",
            min_value=1,
            max_value=5,
            value=DEFAULT_CONFIRMATION_STEPS,
            step=1,
            help="Dùng để phân biệt chuyển nhóm bền hơn với thay đổi ngắn hạn.",
        )
        confidence_threshold = st.slider(
            "Độ chắc chắn tối thiểu khi gán nhóm",
            min_value=0.0,
            max_value=0.9,
            value=DEFAULT_CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Giá trị cao hơn yêu cầu cổ phiếu nằm rõ hơn trong nhóm. Đây là tham số kỹ thuật, có thể giữ mặc định nếu không nghiên cứu sâu.",
        )

    run_model = st.button("Phân tích hành vi", use_container_width=True)

api_key = st.session_state.get("vnstock_api_key", "").strip() or os.getenv("VNSTOCK_API_KEY", "").strip()

if pd.Timestamp(start) >= pd.Timestamp(end):
    st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
    st.stop()

if update_data:
    if not api_key:
        st.error("Chưa có mã truy cập VNstock.")
        st.stop()

    symbols = symbols_for_period(pd.Timestamp(start) - pd.Timedelta(days=136), pd.Timestamp(end))
    total = len(symbols) + 1
    progress = st.progress(0)
    status_box = st.empty()
    detail_box = st.empty()
    error_box = st.empty()
    error_messages = []

    def on_progress(done: int, total_count: int, symbol: str, status: str):
        progress.progress(min(done / total_count, 1.0))
        status_box.info(f"Đang tải {done}/{total_count}: {symbol}")
        detail_box.caption(status)
        if status.startswith("LỖI:"):
            error_messages.append(f"{symbol}: {status[5:].strip()}")
            error_box.warning("Mã chưa tải được: " + " | ".join(error_messages[-5:]))

    try:
        with st.spinner("Đang tải bộ dữ liệu mới. Không sử dụng dữ liệu cũ."):
            stock, index = load_market_data(pd.Timestamp(start), pd.Timestamp(end), api_key=api_key, progress_callback=on_progress)
        st.session_state["market_data"] = (stock, index)
        st.session_state.pop("model_result", None)
        progress.progress(1.0)
        status_box.success(f"Đã tải xong: {len(stock):,} dòng dữ liệu cổ phiếu và {len(index):,} dòng VN Index.")
        detail_box.caption("Dữ liệu hiện nằm trong phiên làm việc. Bạn có thể thử nhiều cách phân nhóm mà không tải lại dữ liệu.")
    except Exception as exc:
        status_box.error("Tải dữ liệu chưa hoàn tất.")
        st.exception(exc)
        st.stop()

if "market_data" not in st.session_state:
    st.info("Bắt đầu bằng cách nhập mã truy cập, chọn khoảng thời gian và bấm Tải dữ liệu mới.")
    st.stop()

stock, index = st.session_state["market_data"]
st.success("Dữ liệu đã sẵn sàng. Bạn có thể thay đổi cách phân nhóm và bấm Phân tích hành vi mà không gọi lại VNstock.")

if run_model:
    try:
        with st.spinner("Đang phân tích cách các cổ phiếu vận động theo thời gian..."):
            feature_panel = build_feature_panel(stock, index)
            rolling_result, diagnostics = rolling_cluster(
                feature_panel,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
                train_window=int(train_window),
                k=int(k),
                step=int(step),
            )
            migration = build_migration_table(
                rolling_result,
                confirmation_steps=int(confirmation_steps),
                confidence_threshold=float(confidence_threshold),
            )
            forward = calculate_forward_returns(stock, migration, horizons=(5, 10, 20))
            forward_summary = summarize_forward_returns(forward)
        st.session_state["model_result"] = (rolling_result, diagnostics, migration, forward_summary)
    except Exception as exc:
        st.error("Không thể hoàn thành phân tích.")
        st.exception(exc)
        st.stop()

if "model_result" in st.session_state:
    rolling_result, diagnostics, migration, forward_summary = st.session_state["model_result"]

    st.subheader("Bản đồ hành vi hiện tại")
    st.caption("Mỗi chấm là một cổ phiếu. Các cổ phiếu nằm gần nhau đang có cách vận động tương tự nhau trong giai đoạn quan sát. Màu sắc thể hiện nhóm hành vi.")
    st.plotly_chart(behavior_map(rolling_result, title="Bản đồ hành vi VN30"), use_container_width=True)

    st.subheader("Các nhóm thay đổi như thế nào?")
    st.caption("Cho biết số cổ phiếu nằm trong từng nhóm ở mỗi lần quan sát. Đây là cách nhìn tổng thể về trạng thái thị trường.")
    st.plotly_chart(cluster_count_chart(rolling_result), use_container_width=True)

    st.subheader("Cổ phiếu chuyển nhóm theo thời gian")
    st.caption("Mỗi hàng là một cổ phiếu. Mỗi cột là một lần quan sát. Màu thay đổi khi hành vi của cổ phiếu chuyển sang nhóm khác. Các mã không thuộc VN30 tại thời điểm đó được để trống.")
    st.plotly_chart(migration_heatmap(migration), use_container_width=True)

    st.subheader("Sau khi chuyển nhóm, lợi suất thay đổi thế nào?")
    st.caption("Bảng này là kiểm tra ban đầu về mối liên hệ giữa trạng thái hành vi và lợi suất các phiên sau. Kết quả chưa đủ để kết luận đây là tín hiệu đầu tư.")
    st.dataframe(forward_summary, use_container_width=True)

    st.subheader("Những lần chuyển nhóm")
    st.caption("Giúp xem cụ thể cổ phiếu nào chuyển nhóm, chuyển vào thời điểm nào và trạng thái mới có được duy trì hay không.")
    display_migration = migration.copy()
    rename_map = {
        "Ticker": "Mã cổ phiếu",
        "Date": "Ngày",
        "ClusterLabel": "Nhóm mới",
        "PreviousObservedCluster": "Nhóm trước",
        "Transition": "Thay đổi",
        "MigrationType": "Loại thay đổi",
        "AssignmentConfidence": "Độ chắc chắn",
        "MigrationSignal": "Tín hiệu chuyển nhóm",
        "MigrationConfirmed": "Chuyển nhóm được xác nhận",
        "FeatureDrivenChange": "Do hành vi thay đổi",
        "ModelDrivenChange": "Do mô hình thay đổi",
    }
    display_migration = display_migration.rename(columns={k: v for k, v in rename_map.items() if k in display_migration.columns})
    preferred = [c for c in ["Mã cổ phiếu", "Ngày", "Nhóm trước", "Nhóm mới", "Thay đổi", "Loại thay đổi", "Độ chắc chắn", "Tín hiệu chuyển nhóm", "Chuyển nhóm được xác nhận"] if c in display_migration.columns]
    st.dataframe(display_migration[preferred], use_container_width=True)

    st.subheader("Thay đổi thành phần VN30")
    st.caption("Chỉ dùng để đánh dấu thời điểm rổ VN30 thay đổi. Số lượng cổ phiếu vẫn là 30; điều quan trọng là mã nào vào và mã nào ra.")
    changes = change_table()
    if changes is not None and not changes.empty:
        st.dataframe(changes, use_container_width=True)
    else:
        st.info("Không có thay đổi thành phần trong khoảng thời gian đã chọn.")

    with st.expander("Thông tin kỹ thuật cho người muốn nghiên cứu sâu"):
        st.write("Sáu nhóm thông tin đầu vào gồm lợi suất 20 phiên, mức biến động 20 phiên, độ nhạy với VN Index trong 60 phiên, sức mạnh tương đối 20 phiên, bất thường khối lượng 20 phiên và mức độ khác biệt so với VN Index trong 60 phiên.")
        st.write("Ứng dụng chuẩn hóa các đặc trưng trước khi phân nhóm K means và liên kết các tâm nhóm giữa các lần quan sát để tránh đổi tên nhóm một cách máy móc.")
        st.write("Độ chắc chắn cho biết một cổ phiếu nằm cách nhóm gần nhất bao xa so với nhóm gần thứ hai. Chuyển nhóm được xác nhận có tính đến việc trạng thái mới được duy trì qua các lần quan sát tiếp theo.")
        st.dataframe(diagnostics, use_container_width=True)
