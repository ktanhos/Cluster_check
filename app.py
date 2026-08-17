import os

import streamlit as st
import pandas as pd

from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIRMATION_STEPS,
    DEFAULT_END,
    DEFAULT_K,
    DEFAULT_START,
    DEFAULT_STEP,
    DEFAULT_TRAIN_WINDOW,
)
from data_loader import load_market_data
from features import build_feature_panel
from clustering import rolling_cluster
from migration import build_migration_table
from backtest import calculate_forward_returns, summarize_forward_returns

st.set_page_config(page_title="VN30 Rolling Behavior Clustering", layout="wide")
st.title("VN30 Rolling Market Behavior Clustering")
st.caption(
    "Rolling clustering trên dữ liệu giá và khối lượng. Feature space được chuẩn hóa theo từng ngày để giữ hình học ổn định giữa các cửa sổ."
)

with st.sidebar:
    st.subheader("Xác thực VNstock")
    st.caption(
        "Nhập API Key VNstock để ứng dụng gọi dữ liệu. Key chỉ được giữ trong phiên chạy hiện tại và không được ghi vào GitHub."
    )
    api_key = st.text_input(
        "VNstock API Key",
        value=st.session_state.get("vnstock_api_key", ""),
        type="password",
        placeholder="Dán API Key VNstock tại đây",
        help="Có thể lấy API Key từ tài khoản Vnstock.",
    )
    if api_key:
        st.session_state["vnstock_api_key"] = api_key

    st.divider()
    start = st.date_input("Ngày bắt đầu", DEFAULT_START)
    end = st.date_input("Ngày kết thúc", DEFAULT_END)
    train_window = st.number_input(
        "Cửa sổ rolling", min_value=60, max_value=504, value=DEFAULT_TRAIN_WINDOW, step=5
    )
    k = st.number_input("Số cụm K", min_value=2, max_value=8, value=DEFAULT_K, step=1)
    step = st.number_input("Bước cập nhật", min_value=1, max_value=20, value=DEFAULT_STEP, step=1)
    confirmation_steps = st.number_input(
        "Số mốc để xác nhận migration",
        min_value=1,
        max_value=5,
        value=DEFAULT_CONFIRMATION_STEPS,
        step=1,
    )
    confidence_threshold = st.slider(
        "Ngưỡng confidence",
        min_value=0.0,
        max_value=0.9,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
    )
    run = st.button("Chạy nghiên cứu", type="primary", use_container_width=True)

if run:
    if pd.Timestamp(start) >= pd.Timestamp(end):
        st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
        st.stop()

    api_key = st.session_state.get("vnstock_api_key", "").strip()
    if not api_key:
        st.error("Chưa có VNstock API Key. Hãy nhập API Key ở thanh bên trước khi chạy.")
        st.stop()

    # Expose the key to vnstock without writing it to the repository or logs.
    os.environ["VNSTOCK_API_KEY"] = api_key

    try:
        with st.spinner("Đang xác thực VNstock, tải dữ liệu và tính toán..."):
            stock, index = load_market_data(
                pd.Timestamp(start), pd.Timestamp(end), api_key=api_key
            )
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
    except Exception as exc:
        st.error("Không thể lấy dữ liệu VNstock hoặc chạy mô hình.")
        st.exception(exc)
        st.info(
            "Nếu lỗi liên quan đến xác thực hoặc giới hạn truy cập, hãy kiểm tra API Key VNstock và thử lại sau một khoảng ngắn. Ứng dụng đã giới hạn số lần gọi API và sử dụng cache để giảm tải."
        )
        st.stop()

    st.success(
        f"Hoàn tất. Có {rolling_result['Date'].nunique()} mốc nghiên cứu và {rolling_result['Ticker'].nunique()} mã."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mốc nghiên cứu", int(rolling_result["Date"].nunique()))
    col2.metric("Migration thô", int(migration["Migration"].sum()))
    col3.metric("Migration xác nhận", int(migration["MigrationConfirmed"].sum()))
    col4.metric("Confidence TB", f"{migration['AssignmentConfidence'].mean():.2f}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Trạng thái", "Migration", "Forward Return", "Chẩn đoán"]
    )

    with tab1:
        latest = rolling_result[rolling_result["Date"] == rolling_result["Date"].max()].copy()
        st.dataframe(latest.sort_values(["Cluster", "Ticker"]), use_container_width=True)

    with tab2:
        st.dataframe(
            migration.sort_values(["Date", "Ticker"]),
            use_container_width=True,
        )
        st.caption(
            "Migration thô ghi nhận mọi thay đổi trạng thái. Migration xác nhận yêu cầu trạng thái mới tồn tại đủ số mốc đã chọn và vượt ngưỡng confidence."
        )

    with tab3:
        st.dataframe(forward_summary, use_container_width=True)
        st.info(
            "Forward Return hiện là event study mô tả. Chưa phải kiểm định alpha hoàn chỉnh và chưa điều chỉnh transaction cost."
        )

    with tab4:
        st.dataframe(diagnostics, use_container_width=True)
        st.caption(
            "Assignment Confidence càng cao thì điểm quan sát càng nằm rõ về một centroid thay vì nằm gần ranh giới giữa hai trạng thái."
        )

    st.download_button(
        "Tải toàn bộ trạng thái CSV",
        rolling_result.to_csv(index=False).encode("utf-8-sig"),
        "rolling_clusters.csv",
        "text/csv",
    )
    st.download_button(
        "Tải Migration CSV",
        migration.to_csv(index=False).encode("utf-8-sig"),
        "migration.csv",
        "text/csv",
    )
else:
    st.info("Nhập API Key VNstock ở thanh bên, chọn khoảng thời gian và tham số rồi bấm Chạy nghiên cứu.")
