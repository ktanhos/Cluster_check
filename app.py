import os

import pandas as pd
import streamlit as st

APP_VERSION = "DATA-LAYER-TEST-2026-08-17-01"

try:
    from vnstock.config import Config
    Config.REQUEST_TIMEOUT = 15
    Config.RETRIES = 1
    Config.BACKOFF_MIN = 1
    Config.BACKOFF_MAX = 3
except Exception:
    pass

from backtest import calculate_forward_returns, summarize_forward_returns
from charts import behavior_map, cluster_count_chart, membership_count_chart, migration_heatmap
from clustering import rolling_cluster
from config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_CONFIRMATION_STEPS, DEFAULT_END, DEFAULT_K, DEFAULT_START, DEFAULT_STEP, DEFAULT_TRAIN_WINDOW
from data_loader import load_market_data
from features import build_feature_panel
from membership import change_table, symbols_for_period
from migration import build_migration_table

st.set_page_config(page_title="VN30 Rolling Behavior Clustering", layout="wide")
st.title("VN30 Rolling Market Behavior Clustering")
st.caption("Lớp dữ liệu và lớp mô hình được tách riêng. VNstock chỉ được gọi khi bấm Cập nhật dữ liệu. Sau đó có thể chạy nhiều cấu hình mô hình mà không gọi API lại.")
st.caption(f"Phiên bản Data Layer: {APP_VERSION}")

with st.sidebar:
    st.subheader("Xác thực VNstock")
    api_key = st.text_input("VNstock API Key", value=st.session_state.get("vnstock_api_key", ""), type="password", placeholder="Dán API Key VNstock tại đây")
    if api_key:
        st.session_state["vnstock_api_key"] = api_key
    st.caption("API Key chỉ được giữ trong phiên Streamlit.")
    st.divider()
    st.subheader("Khoảng dữ liệu")
    start = st.date_input("Ngày bắt đầu", DEFAULT_START)
    end = st.date_input("Ngày kết thúc", DEFAULT_END)
    update_data = st.button("Cập nhật dữ liệu VNstock", type="primary", use_container_width=True)
    clear_data = st.button("Xóa dữ liệu cache", use_container_width=True, help="Xóa dữ liệu đã lưu trong phiên Cloud. Chỉ dùng khi muốn tải lại từ đầu.")
    st.divider()
    st.subheader("Thiết lập mô hình")
    train_window = st.number_input("Cửa sổ rolling", min_value=60, max_value=504, value=DEFAULT_TRAIN_WINDOW, step=5)
    k = st.number_input("Số cụm K", min_value=2, max_value=8, value=DEFAULT_K, step=1)
    step = st.number_input("Bước cập nhật", min_value=1, max_value=20, value=DEFAULT_STEP, step=1)
    confirmation_steps = st.number_input("Số mốc xác nhận migration", min_value=1, max_value=5, value=DEFAULT_CONFIRMATION_STEPS, step=1)
    confidence_threshold = st.slider("Ngưỡng confidence", min_value=0.0, max_value=0.9, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05)
    run_model = st.button("Chạy mô hình", use_container_width=True)

api_key = st.session_state.get("vnstock_api_key", "").strip() or os.getenv("VNSTOCK_API_KEY", "").strip()

if pd.Timestamp(start) >= pd.Timestamp(end):
    st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
    st.stop()

if clear_data:
    from data_loader import CACHE_DIR
    for p in CACHE_DIR.glob("*.csv"):
        try:
            p.unlink()
        except OSError:
            pass
    st.session_state.pop("market_data", None)
    st.session_state.pop("model_result", None)
    st.success("Đã xóa cache dữ liệu. Chưa có API nào được gọi.")

if update_data:
    if not api_key:
        st.error("Chưa có VNstock API Key.")
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
        status_box.info(f"Đang xử lý {done}/{total_count}: {symbol}")
        detail_box.caption(status)
        if status.startswith("LỖI:"):
            error_messages.append(f"{symbol}: {status[5:].strip()}")
            error_box.warning("Mã chưa tải được: " + " | ".join(error_messages[-5:]))

    try:
        with st.spinner("Đang tải dữ liệu VNstock. Mỗi mã được xử lý độc lập; mã lỗi không làm treo toàn bộ danh sách."):
            stock, index = load_market_data(
                pd.Timestamp(start),
                pd.Timestamp(end),
                api_key=api_key,
                progress_callback=on_progress,
                force_refresh=False,
            )
        st.session_state["market_data"] = (stock, index)
        st.session_state.pop("model_result", None)
        progress.progress(1.0)
        status_box.success(f"Đã sẵn sàng dữ liệu: {len(stock):,} dòng cổ phiếu và {len(index):,} dòng VNINDEX.")
        detail_box.caption("Dữ liệu đã được lưu theo từng mã. Chạy mô hình không gọi VNstock.")
    except Exception as exc:
        status_box.error("Cập nhật dữ liệu chưa hoàn tất.")
        st.exception(exc)
        st.warning("Các mã tải thành công vẫn được giữ lại. Bấm Cập nhật dữ liệu lần nữa để tiếp tục; hệ thống sẽ dùng cache và chỉ tải phần còn thiếu.")
        st.stop()

if "market_data" not in st.session_state:
    st.info("Nhập API Key, chọn khoảng thời gian rồi bấm Cập nhật dữ liệu VNstock. Chưa bấm nút này thì ứng dụng không gọi VNstock.")
    st.stop()

stock, index = st.session_state["market_data"]
st.success("Dữ liệu đã sẵn sàng. Thay đổi K, cửa sổ rolling hoặc bước cập nhật không gọi lại VNstock.")

if run_model:
    try:
        with st.spinner("Đang tính feature, rolling clustering, migration và forward return..."):
            feature_panel = build_feature_panel(stock, index)
            rolling_result, diagnostics = rolling_cluster(feature_panel, start=pd.Timestamp(start), end=pd.Timestamp(end), train_window=int(train_window), k=int(k), step=int(step))
            migration = build_migration_table(rolling_result, confirmation_steps=int(confirmation_steps), confidence_threshold=float(confidence_threshold))
            forward = calculate_forward_returns(stock, migration, horizons=(5, 10, 20))
            forward_summary = summarize_forward_returns(forward)
        st.session_state["model_result"] = (rolling_result, diagnostics, migration, forward_summary)
    except Exception as exc:
        st.error("Không thể chạy mô hình.")
        st.exception(exc)
        st.stop()

if "model_result" in st.session_state:
    rolling_result, diagnostics, migration, forward_summary = st.session_state["model_result"]
    st.subheader("Behavior Map")
    st.plotly_chart(behavior_map(rolling_result), use_container_width=True)
    st.subheader("Cluster Timeline")
    st.plotly_chart(cluster_count_chart(rolling_result), use_container_width=True)
    st.subheader("Migration Heatmap")
    st.plotly_chart(migration_heatmap(migration), use_container_width=True)
    st.subheader("Thành phần VN30")
    st.plotly_chart(membership_count_chart(pd.Timestamp(start), pd.Timestamp(end)), use_container_width=True)
    st.subheader("Forward Return")
    st.dataframe(forward_summary, use_container_width=True)
    st.subheader("Migration Detail")
    st.dataframe(migration, use_container_width=True)
    st.subheader("Thay đổi thành phần")
    st.dataframe(change_table(), use_container_width=True)
