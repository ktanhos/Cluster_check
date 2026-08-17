import streamlit as st
import pandas as pd
import numpy as np

from config import VN30, DEFAULT_START, DEFAULT_END, DEFAULT_TRAIN_WINDOW, DEFAULT_K, DEFAULT_STEP
from data_loader import load_market_data
from features import build_feature_panel, get_feature_snapshot
from clustering import rolling_cluster
from migration import build_migration_table
from backtest import calculate_forward_returns, summarize_forward_returns

st.set_page_config(page_title="VN30 Rolling Behavior Clustering", layout="wide")
st.title("VN30 Rolling Market Behavior Clustering")
st.caption("Rolling clustering trên dữ liệu giá và khối lượng, với dữ liệu lịch sử tải từ VNstock và lưu cache cục bộ.")

with st.sidebar:
    start = st.date_input("Ngày bắt đầu", DEFAULT_START)
    end = st.date_input("Ngày kết thúc", DEFAULT_END)
    train_window = st.number_input("Cửa sổ rolling", min_value=60, max_value=504, value=DEFAULT_TRAIN_WINDOW, step=5)
    k = st.number_input("Số cụm K", min_value=2, max_value=8, value=DEFAULT_K, step=1)
    step = st.number_input("Bước cập nhật", min_value=1, max_value=20, value=DEFAULT_STEP, step=1)
    run = st.button("Chạy nghiên cứu", type="primary", use_container_width=True)

if run:
    with st.spinner("Đang tải dữ liệu và tính toán..."):
        stock, index = load_market_data(pd.Timestamp(start), pd.Timestamp(end))
        feature_panel = build_feature_panel(stock, index)
        rolling_result, diagnostics = rolling_cluster(
            feature_panel,
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            train_window=int(train_window),
            k=int(k),
            step=int(step),
        )
        migration = build_migration_table(rolling_result)
        forward = calculate_forward_returns(stock, migration, horizons=(5, 10, 20))
        forward_summary = summarize_forward_returns(forward)

    st.success(f"Hoàn tất. Có {rolling_result['Date'].nunique()} mốc nghiên cứu và {rolling_result['Ticker'].nunique()} mã.")

    tab1, tab2, tab3, tab4 = st.tabs(["Trạng thái", "Migration", "Forward Return", "Chẩn đoán"])

    with tab1:
        latest = rolling_result[rolling_result["Date"] == rolling_result["Date"].max()].copy()
        st.dataframe(latest.sort_values(["Cluster", "Ticker"]), use_container_width=True)

    with tab2:
        st.dataframe(migration.sort_values(["Date", "Ticker"]), use_container_width=True)

    with tab3:
        st.dataframe(forward_summary, use_container_width=True)
        st.info("Forward Return ở đây là thống kê sự kiện ban đầu, chưa phải kiểm định alpha hoàn chỉnh.")

    with tab4:
        st.dataframe(diagnostics, use_container_width=True)

    st.download_button("Tải toàn bộ trạng thái CSV", rolling_result.to_csv(index=False).encode("utf-8-sig"), "rolling_clusters.csv", "text/csv")
    st.download_button("Tải Migration CSV", migration.to_csv(index=False).encode("utf-8-sig"), "migration.csv", "text/csv")
else:
    st.info("Chọn khoảng thời gian và tham số ở thanh bên rồi bấm Chạy nghiên cứu.")
