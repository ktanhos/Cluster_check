import os

import pandas as pd
import streamlit as st

from backtest import calculate_forward_returns, summarize_forward_returns
from charts import behavior_map, cluster_count_chart, membership_count_chart, migration_heatmap
from clustering import rolling_cluster
from config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_CONFIRMATION_STEPS, DEFAULT_END, DEFAULT_K, DEFAULT_START, DEFAULT_STEP, DEFAULT_TRAIN_WINDOW
from data_loader import cache_status, load_market_data
from features import build_feature_panel
from membership import change_table, symbols_for_period
from migration import build_migration_table

st.set_page_config(page_title="VN30 Rolling Behavior Clustering", layout="wide")
st.title("VN30 Rolling Market Behavior Clustering")
st.caption("Tách riêng lớp dữ liệu và lớp mô hình. Dữ liệu VNstock chỉ tải khi bấm cập nhật dữ liệu; sau đó có thể chạy nhiều cấu hình mô hình mà không gọi API lại.")

with st.sidebar:
    st.subheader("Xác thực VNstock")
    api_key = st.text_input("VNstock API Key", value=st.session_state.get("vnstock_api_key", ""), type="password", placeholder="Dán API Key VNstock tại đây")
    if api_key:
        st.session_state["vnstock_api_key"] = api_key
    st.caption("API Key chỉ được giữ trong phiên Streamlit. Có thể dùng Streamlit Secrets với khóa VNSTOCK_API_KEY.")
    st.divider()
    st.subheader("Khoảng dữ liệu")
    start = st.date_input("Ngày bắt đầu", DEFAULT_START)
    end = st.date_input("Ngày kết thúc", DEFAULT_END)
    force_refresh = st.checkbox("Bắt buộc tải lại từ VNstock", value=False)
    update_data = st.button("Cập nhật dữ liệu VNstock", type="primary", use_container_width=True)
    st.divider()
    st.subheader("Thiết lập mô hình")
    train_window = st.number_input("Cửa sổ rolling", min_value=60, max_value=504, value=DEFAULT_TRAIN_WINDOW, step=5)
    k = st.number_input("Số cụm K", min_value=2, max_value=8, value=DEFAULT_K, step=1)
    step = st.number_input("Bước cập nhật", min_value=1, max_value=20, value=DEFAULT_STEP, step=1)
    confirmation_steps = st.number_input("Số mốc xác nhận migration", min_value=1, max_value=5, value=DEFAULT_CONFIRMATION_STEPS, step=1)
    confidence_threshold = st.slider("Ngưỡng confidence", min_value=0.0, max_value=0.9, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05)
    run_model = st.button("Chạy mô hình", use_container_width=True)
    if st.button("Xóa cache Streamlit", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop("market_data", None)
        st.success("Đã xóa cache phiên.")

api_key = st.session_state.get("vnstock_api_key", "").strip() or os.getenv("VNSTOCK_API_KEY", "").strip()

if pd.Timestamp(start) >= pd.Timestamp(end):
    st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
    st.stop()

if update_data:
    if not api_key:
        st.error("Chưa có VNstock API Key.")
        st.stop()
    progress = st.progress(0)
    status_box = st.empty()
    detail_box = st.empty()
    try:
        symbols = symbols_for_period(pd.Timestamp(start) - pd.Timedelta(days=136), pd.Timestamp(end))
        total = len(symbols) + 1
        status_box.info(f"Chuẩn bị dữ liệu cho {len(symbols)} mã có thể thuộc VN30 trong khoảng nghiên cứu, cộng VNINDEX.")

        def on_progress(done: int, total_count: int, symbol: str, status: str):
            progress.progress(min(done / total_count, 1.0))
            status_box.info(f"Đang xử lý {done}/{total_count}: {symbol}")
            detail_box.caption(status)

        with st.spinner("Đang cập nhật dữ liệu. Lần đầu có thể mất vài phút do giới hạn tốc độ API..."):
            stock, index = load_market_data(pd.Timestamp(start), pd.Timestamp(end), api_key=api_key, progress_callback=on_progress, force_refresh=force_refresh)
        st.session_state["market_data"] = (stock, index)
        progress.progress(1.0)
        status_box.success(f"Đã sẵn sàng dữ liệu: {len(stock):,} dòng cổ phiếu và {len(index):,} dòng VNINDEX.")
        detail_box.caption("Dữ liệu đã được lưu cache theo từng mã. Lần chạy sau chỉ tải phần còn thiếu.")
    except Exception as exc:
        status_box.error("Cập nhật dữ liệu thất bại.")
        st.exception(exc)
        st.stop()

if "market_data" not in st.session_state:
    st.info("Nhập API Key, chọn khoảng thời gian rồi bấm Cập nhật dữ liệu VNstock. Sau khi dữ liệu sẵn sàng mới bấm Chạy mô hình.")
    st.stop()

stock, index = st.session_state["market_data"]

st.success("Dữ liệu đã sẵn sàng. Thay đổi K, cửa sổ rolling hoặc bước cập nhật không cần gọi lại VNstock.")

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

if "model_result" not in st.session_state:
    st.info("Bấm Chạy mô hình để bắt đầu rolling clustering.")
    st.stop()

rolling_result, diagnostics, migration, forward_summary = st.session_state["model_result"]
n_dates = int(rolling_result["Date"].nunique())
feature_migrations = int(migration["EconomicallyDrivenMigration"].sum())
model_migrations = int((migration["MigrationType"] == "Model-driven").sum())
mixed_migrations = int((migration["MigrationType"] == "Mixed").sum())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Mốc nghiên cứu", n_dates)
col2.metric("Migration", int(migration["MigrationSignal"].sum()))
col3.metric("Feature-driven", feature_migrations)
col4.metric("Model-driven", model_migrations)
col5.metric("Mixed", mixed_migrations)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Behavior Map", "Migration", "Thành phần VN30", "Forward Return", "Chẩn đoán"])
with tab1:
    latest_date = rolling_result["Date"].max()
    latest = rolling_result[rolling_result["Date"] == latest_date].copy()
    st.plotly_chart(behavior_map(latest, f"Biểu đồ hành vi VN30 tại {latest_date:%d/%m/%Y}"), use_container_width=True)
    st.plotly_chart(cluster_count_chart(rolling_result), use_container_width=True)
    st.dataframe(latest.sort_values(["Cluster", "Ticker"]), use_container_width=True)
with tab2:
    st.plotly_chart(migration_heatmap(migration, change_table()), use_container_width=True)
    st.dataframe(migration[migration["MigrationSignal"]].sort_values(["Date", "Ticker"]), use_container_width=True)
with tab3:
    st.subheader("Lịch sử thay đổi thành phần")
    st.dataframe(change_table(), use_container_width=True)
    st.plotly_chart(membership_count_chart(sorted(pd.to_datetime(rolling_result["Date"].unique()))), use_container_width=True)
with tab4:
    st.dataframe(forward_summary, use_container_width=True)
with tab5:
    st.dataframe(diagnostics, use_container_width=True)
    diagnostic_cols = [c for c in ["Date", "Ticker", "Transition", "MigrationType", "CentroidDrift", "AssignmentConfidence", "PreviousObservedCluster", "PreviousModelCluster"] if c in migration.columns]
    st.dataframe(migration[migration["MigrationSignal"]][diagnostic_cols].sort_values("Date"), use_container_width=True)

st.download_button("Tải trạng thái cụm CSV", rolling_result.to_csv(index=False).encode("utf-8-sig"), "rolling_clusters.csv", "text/csv")
st.download_button("Tải Migration CSV", migration.to_csv(index=False).encode("utf-8-sig"), "migration.csv", "text/csv")
