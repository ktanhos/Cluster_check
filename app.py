import os

import pandas as pd
import streamlit as st

from backtest import calculate_forward_returns, summarize_forward_returns
from charts import behavior_map, cluster_count_chart, membership_count_chart, migration_heatmap
from clustering import rolling_cluster
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
from membership import change_table
from migration import build_migration_table

st.set_page_config(page_title="VN30 Rolling Behavior Clustering", layout="wide")
st.title("VN30 Rolling Market Behavior Clustering")
st.caption("Rolling clustering trên giá và khối lượng, có điều chỉnh theo lịch sử thành phần rổ VN30.")

with st.sidebar:
    st.subheader("Xác thực VNstock")
    api_key = st.text_input(
        "VNstock API Key",
        value=st.session_state.get("vnstock_api_key", ""),
        type="password",
        placeholder="Dán API Key VNstock tại đây",
        help="API Key chỉ được giữ trong phiên chạy và không được ghi vào GitHub.",
    )
    if api_key:
        st.session_state["vnstock_api_key"] = api_key
    st.caption("Có thể dùng Streamlit Secrets với khóa VNSTOCK_API_KEY thay cho ô nhập.")

    st.divider()
    st.subheader("Thiết lập nghiên cứu")
    start = st.date_input("Ngày bắt đầu", DEFAULT_START)
    end = st.date_input("Ngày kết thúc", DEFAULT_END)
    train_window = st.number_input("Cửa sổ rolling", min_value=60, max_value=504, value=DEFAULT_TRAIN_WINDOW, step=5)
    k = st.number_input("Số cụm K", min_value=2, max_value=8, value=DEFAULT_K, step=1)
    step = st.number_input("Bước cập nhật", min_value=1, max_value=20, value=DEFAULT_STEP, step=1)
    confirmation_steps = st.number_input("Số mốc để xác nhận migration", min_value=1, max_value=5, value=DEFAULT_CONFIRMATION_STEPS, step=1)
    confidence_threshold = st.slider("Ngưỡng confidence", min_value=0.0, max_value=0.9, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05)
    run = st.button("Chạy nghiên cứu", type="primary", use_container_width=True)

if run:
    if pd.Timestamp(start) >= pd.Timestamp(end):
        st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
        st.stop()

    api_key = st.session_state.get("vnstock_api_key", "").strip() or os.getenv("VNSTOCK_API_KEY", "").strip()
    if not api_key:
        st.error("Chưa có VNstock API Key. Hãy nhập API Key ở thanh bên hoặc cấu hình VNSTOCK_API_KEY trong Streamlit Secrets.")
        st.stop()

    try:
        with st.spinner("Đang tải dữ liệu, tính feature và chạy rolling clustering..."):
            stock, index = load_market_data(pd.Timestamp(start), pd.Timestamp(end), api_key=api_key)
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
        st.info("Kiểm tra API Key, giới hạn truy cập và dữ liệu cache rồi thử lại.")
        st.stop()

    n_dates = int(rolling_result["Date"].nunique())
    st.success(f"Hoàn tất. Có {n_dates} mốc nghiên cứu; mỗi mốc chỉ sử dụng các mã đang thuộc VN30 tại đúng ngày đó.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mốc nghiên cứu", n_dates)
    col2.metric("Migration thô", int(migration["Migration"].sum()))
    col3.metric("Migration xác nhận", int(migration["MigrationConfirmed"].sum()))
    col4.metric("Confidence TB", f"{migration['AssignmentConfidence'].mean():.2f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Behavior Map", "Migration", "Thành phần VN30", "Forward Return", "Chẩn đoán"])

    with tab1:
        latest_date = rolling_result["Date"].max()
        latest = rolling_result[rolling_result["Date"] == latest_date].copy()
        st.plotly_chart(
            behavior_map(latest, f"Biểu đồ hành vi VN30 tại {latest_date:%d/%m/%Y}"),
            use_container_width=True,
        )
        st.caption("Behavior Map chỉ vẽ các mã đang là thành phần VN30 tại ngày quan sát cuối. Tâm cụm trên hình là tâm thực nghiệm của các cổ phiếu được gán vào từng nhóm trong không gian hai chiều.")
        st.plotly_chart(cluster_count_chart(rolling_result), use_container_width=True)
        st.dataframe(latest.sort_values(["Cluster", "Ticker"]), use_container_width=True)

    with tab2:
        st.plotly_chart(migration_heatmap(migration, change_table()), use_container_width=True)
        st.caption("Ô trống có chủ đích: mã chưa thuộc VN30 tại thời điểm đó hoặc chưa có kết quả phân loại. Đường đỏ là ngày hiệu lực thay đổi thành phần; chú thích phía trên cho biết mã vào và mã ra.")
        st.dataframe(migration.sort_values(["Date", "Ticker"]), use_container_width=True)

    with tab3:
        st.subheader("Lịch sử thay đổi thành phần")
        st.dataframe(change_table(), use_container_width=True)
        st.plotly_chart(
            membership_count_chart(sorted(pd.to_datetime(rolling_result["Date"].unique()))),
            use_container_width=True,
        )
        st.info("Rolling clustering chỉ sử dụng các mã đang thuộc rổ VN30 tại từng ngày. Mã bị loại dừng tham gia từ ngày hiệu lực; mã mới bắt đầu tham gia từ ngày hiệu lực.")

    with tab4:
        st.dataframe(forward_summary, use_container_width=True)
        st.info("Forward Return hiện là event study mô tả. Chưa phải kiểm định alpha hoàn chỉnh và chưa điều chỉnh chi phí giao dịch.")

    with tab5:
        st.dataframe(diagnostics, use_container_width=True)
        st.caption("Confidence cao nghĩa là điểm quan sát nằm rõ gần centroid của trạng thái được gán; confidence thấp nghĩa là gần ranh giới giữa các trạng thái.")

    st.download_button("Tải toàn bộ trạng thái CSV", rolling_result.to_csv(index=False).encode("utf-8-sig"), "rolling_clusters.csv", "text/csv")
    st.download_button("Tải Migration CSV", migration.to_csv(index=False).encode("utf-8-sig"), "migration.csv", "text/csv")
else:
    st.info("Nhập API Key VNstock ở thanh bên, chọn khoảng thời gian và tham số rồi bấm Chạy nghiên cứu.")
