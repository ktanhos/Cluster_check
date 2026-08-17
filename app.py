import os

import pandas as pd
import streamlit as st

APP_VERSION = "USER-FRIENDLY-2026-08-17-04"

from backtest import build_reference_forecast, calculate_forward_returns, evaluate_reference_forecast, summarize_forward_returns
from charts import behavior_history_chart, behavior_map, behavior_trajectory_chart, cluster_count_chart, migration_heatmap
from clustering import rolling_cluster
from config import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_CONFIRMATION_STEPS, DEFAULT_END, DEFAULT_K, DEFAULT_START, DEFAULT_STEP, DEFAULT_TRAIN_WINDOW, VN30
from data_loader import load_market_data
from features import build_feature_panel
from membership import change_table, symbols_for_period
from migration import build_migration_table

st.set_page_config(page_title="VN30 Hành vi thị trường", layout="wide")
st.title("VN30 Hành vi thị trường")
st.caption("Ứng dụng nhóm các cổ phiếu VN30 theo cách chúng đang vận động trên thị trường, chỉ dựa trên giá và khối lượng. Đây là công cụ nghiên cứu, không phải công cụ khuyến nghị mua bán.")

with st.sidebar:
    st.header("1. Dữ liệu")
    api_key = st.text_input("Mã truy cập VNstock", value=st.session_state.get("vnstock_api_key", ""), type="password", placeholder="Dán mã truy cập VNstock")
    if api_key:
        st.session_state["vnstock_api_key"] = api_key
    start = st.date_input("Từ ngày", DEFAULT_START)
    end = st.date_input("Đến ngày", DEFAULT_END)
    update_data = st.button("Tải dữ liệu mới", type="primary", use_container_width=True)
    st.caption("Chỉ khi bấm nút này ứng dụng mới gọi VNstock. Sau khi tải xong, thay đổi thiết lập bên dưới không gọi lại dữ liệu.")

    st.divider()
    st.header("2. Cách nhận diện nhóm")
    train_window = st.number_input("Số phiên dùng để nhận diện hành vi", min_value=60, max_value=504, value=DEFAULT_TRAIN_WINDOW, step=5, help="Số phiên giao dịch gần nhất dùng để nhận diện trạng thái.")
    k = st.number_input("Số nhóm hành vi", min_value=2, max_value=8, value=DEFAULT_K, step=1, help="Số nhóm cổ phiếu được hình thành. Mặc định là 4.")
    step = st.number_input("Cứ bao nhiêu phiên cập nhật một lần", min_value=1, max_value=20, value=DEFAULT_STEP, step=1, help="Ví dụ 5 nghĩa là cứ 5 phiên tạo một lần quan sát mới.")
    with st.expander("Thiết lập nâng cao", expanded=False):
        confirmation_steps = st.number_input("Một lần chuyển nhóm cần được giữ trong bao nhiêu lần quan sát", min_value=1, max_value=5, value=DEFAULT_CONFIRMATION_STEPS, step=1)
        confidence_threshold = st.slider("Độ chắc chắn tối thiểu khi gán nhóm", min_value=0.0, max_value=0.9, value=DEFAULT_CONFIDENCE_THRESHOLD, step=0.05)
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
            rolling_result, diagnostics = rolling_cluster(feature_panel, start=pd.Timestamp(start), end=pd.Timestamp(end), train_window=int(train_window), k=int(k), step=int(step))
            migration = build_migration_table(rolling_result, confirmation_steps=int(confirmation_steps), confidence_threshold=float(confidence_threshold))
            forward = calculate_forward_returns(stock, migration, horizons=(5, 10, 20))
            forward_summary = summarize_forward_returns(forward)
            reference_forecast = build_reference_forecast(stock, rolling_result, horizons=(5, 10, 20), min_history=5)
            forecast_detail, forecast_validation = evaluate_reference_forecast(stock, rolling_result, horizons=(5, 10, 20), min_history=5)
        st.session_state["model_result"] = (rolling_result, diagnostics, migration, forward, forward_summary, reference_forecast, forecast_detail, forecast_validation)
    except Exception as exc:
        st.error("Không thể hoàn thành phân tích.")
        st.exception(exc)
        st.stop()

if "model_result" in st.session_state:
    stored = st.session_state["model_result"]
    if len(stored) == 6:
        rolling_result, diagnostics, migration, forward, forward_summary, reference_forecast = stored
        forecast_detail, forecast_validation = pd.DataFrame(), pd.DataFrame()
    else:
        rolling_result, diagnostics, migration, forward, forward_summary, reference_forecast, forecast_detail, forecast_validation = stored

    st.subheader("Bản đồ hành vi hiện tại")
    st.caption("Ảnh chụp trạng thái mới nhất. Mỗi chấm là một cổ phiếu đang thuộc VN30 tại ngày cuối cùng.")
    st.plotly_chart(behavior_map(rolling_result), use_container_width=True)

    st.subheader("Theo dõi cổ phiếu")
    st.caption("Chọn một mã để xem toàn bộ hành trình; chọn nhiều mã để so sánh thời điểm chuyển nhóm.")
    available_tickers = sorted(set(VN30).intersection(set(rolling_result["Ticker"].unique())))
    selected_tickers = st.multiselect("Chọn cổ phiếu muốn theo dõi", options=available_tickers, default=available_tickers[:1], max_selections=10)
    if selected_tickers:
        st.plotly_chart(behavior_history_chart(rolling_result, selected_tickers, start=pd.Timestamp(start), end=pd.Timestamp(end)), use_container_width=True)
        if len(selected_tickers) == 1:
            st.plotly_chart(behavior_trajectory_chart(rolling_result, selected_tickers[0]), use_container_width=True)
            st.info("Đi sang phải nghĩa là sức mạnh tương đối tăng; đi lên nghĩa là mức biến động tăng. Đường nối cho thấy cổ phiếu đã di chuyển giữa các trạng thái như thế nào.")

    st.subheader("Các nhóm thay đổi như thế nào?")
    st.caption("Số cổ phiếu trong từng nhóm ở mỗi lần quan sát.")
    st.plotly_chart(cluster_count_chart(rolling_result), use_container_width=True)

    st.subheader("Cổ phiếu chuyển nhóm theo thời gian")
    st.caption("Mỗi hàng là một cổ phiếu. Các ô màu cho biết cổ phiếu đang thuộc nhóm nào. Vạch đỏ đánh dấu thời điểm thành phần VN30 thay đổi.")
    st.plotly_chart(migration_heatmap(migration), use_container_width=True)

    st.subheader("Kiểm tra 1: chuyển nhóm có liên quan đến lợi suất không?")
    st.caption("Nhìn lại các lần chuyển nhóm trong quá khứ và xem lợi suất 5, 10, 20 phiên sau đó. Đây là kiểm tra quan hệ lịch sử, chưa phải dự báo.")
    if forward_summary.empty:
        st.info("Chưa có đủ lần chuyển nhóm để thực hiện kiểm tra này.")
    else:
        display_forward = forward_summary.copy()
        for c in ["ForwardReturn5D_Mean", "ForwardReturn5D_Median", "ForwardReturn10D_Mean", "ForwardReturn20D_Mean", "ForwardReturn5D_PositiveRate"]:
            if c in display_forward:
                display_forward[c] = display_forward[c] * 100
        display_forward = display_forward.rename(columns={
            "Transition": "Chuyển từ nhóm nào sang nhóm nào", "Events": "Số lần xuất hiện", "ForwardReturn5D_EventsInBasket": "Số quan sát 5 phiên", "ForwardReturn5D_Mean": "Lợi suất TB sau 5 phiên", "ForwardReturn5D_Median": "Trung vị sau 5 phiên", "ForwardReturn5D_PositiveRate": "Tỷ lệ tăng sau 5 phiên", "ForwardReturn10D_Mean": "Lợi suất TB sau 10 phiên", "ForwardReturn20D_Mean": "Lợi suất TB sau 20 phiên"})
        keep = [c for c in ["Chuyển từ nhóm nào sang nhóm nào", "Số lần xuất hiện", "Số quan sát 5 phiên", "Lợi suất TB sau 5 phiên", "Trung vị sau 5 phiên", "Tỷ lệ tăng sau 5 phiên", "Lợi suất TB sau 10 phiên", "Lợi suất TB sau 20 phiên"] if c in display_forward.columns]
        st.dataframe(display_forward[keep], use_container_width=True)

    st.subheader("Kiểm tra 2: trạng thái hiện tại gợi ý điều gì nếu nhìn lại lịch sử?")
    st.caption("Ứng dụng lấy các quan sát trong quá khứ có cùng nhóm hành vi. Nếu có đủ dữ liệu về đúng kiểu chuyển nhóm, bằng chứng đó được ưu tiên. Đây mới là xếp hạng tham khảo.")
    if reference_forecast.empty:
        st.info("Chưa đủ dữ liệu lịch sử để tạo xếp hạng tham khảo.")
    else:
        display_forecast = reference_forecast.copy()
        percentage_cols = [c for c in ["ReferenceScore", "HistoricalMean5D", "HistoricalMean10D", "HistoricalMean20D", "TransitionMean5D", "TransitionMean10D", "TransitionMean20D", "PositiveRate5D", "PositiveRate10D", "PositiveRate20D"] if c in display_forecast.columns]
        for c in percentage_cols:
            display_forecast[c] = display_forecast[c] * 100
        display_forecast = display_forecast.rename(columns={"Rank": "Xếp hạng tham khảo", "Ticker": "Mã cổ phiếu", "CurrentGroup": "Nhóm hiện tại", "CurrentStatus": "Trạng thái hiện tại", "CurrentTransition": "Chuyển nhóm hiện tại", "Confidence": "Độ chắc chắn", "HistoricalObservations": "Số quan sát cùng nhóm", "TransitionObservations": "Số quan sát cùng kiểu chuyển", "HistoricalMean5D": "TB sau 5 phiên cùng nhóm", "HistoricalMean10D": "TB sau 10 phiên cùng nhóm", "HistoricalMean20D": "TB sau 20 phiên cùng nhóm", "TransitionMean5D": "TB sau 5 phiên cùng kiểu chuyển", "TransitionMean10D": "TB sau 10 phiên cùng kiểu chuyển", "TransitionMean20D": "TB sau 20 phiên cùng kiểu chuyển", "PositiveRate5D": "Tỷ lệ tăng 5 phiên", "ReferenceScore": "Điểm tham khảo", "ForecastBasis": "Căn cứ xếp hạng", "EnoughHistory": "Đủ dữ liệu lịch sử"})
        keep = [c for c in ["Xếp hạng tham khảo", "Mã cổ phiếu", "Nhóm hiện tại", "Trạng thái hiện tại", "Chuyển nhóm hiện tại", "Độ chắc chắn", "Số quan sát cùng nhóm", "Số quan sát cùng kiểu chuyển", "TB sau 5 phiên cùng nhóm", "TB sau 10 phiên cùng nhóm", "TB sau 20 phiên cùng nhóm", "TB sau 5 phiên cùng kiểu chuyển", "Điểm tham khảo", "Căn cứ xếp hạng", "Đủ dữ liệu lịch sử"] if c in display_forecast.columns]
        st.dataframe(display_forecast[keep], use_container_width=True)
        st.warning("Xếp hạng này chưa phải mô hình dự báo được kiểm định ngoài mẫu.")

    st.subheader("Kiểm tra 3: xếp hạng có dự báo được lợi suất ngoài mẫu không?")
    st.caption("Đây là bước quan trọng hơn bảng tham khảo. Tại mỗi ngày trong quá khứ, hệ thống chỉ dùng thông tin đã có trước ngày đó để tạo xếp hạng. Sau đó mới so sánh với lợi suất thực tế 5, 10 và 20 phiên tiếp theo. Vì vậy không dùng dữ liệu tương lai để tạo dự báo.")
    if forecast_validation.empty:
        st.info("Chưa đủ số lần quan sát để thực hiện kiểm định ngoài mẫu. Hãy dùng khoảng thời gian dài hơn hoặc giảm bước quan sát.")
    else:
        display_validation = forecast_validation.copy()
        for c in ["MeanSpearmanIC", "DirectionalAccuracy", "MeanTopBottomSpread", "MAE"]:
            if c in display_validation:
                display_validation[c] = display_validation[c] * 100
        display_validation = display_validation.rename(columns={"Horizon": "Kỳ dự báo", "Method": "Cách dự báo", "ForecastDates": "Số ngày kiểm định", "AverageObservations": "Số cổ phiếu TB", "MeanSpearmanIC": "Tương quan xếp hạng TB", "PositiveICRate": "Tỷ lệ ngày tương quan dương", "DirectionalAccuracy": "Tỷ lệ dự báo đúng hướng", "MeanTopBottomSpread": "Chênh lệch nhóm cao và thấp", "MAE": "Sai số tuyệt đối TB"})
        st.dataframe(display_validation, use_container_width=True)

        comparison = forecast_validation.pivot(index="Horizon", columns="Method", values="MeanSpearmanIC")
        if "StateOnly" in comparison.columns and "MigrationAware" in comparison.columns:
            comparison["MigrationGain"] = comparison["MigrationAware"] - comparison["StateOnly"]
            display_gain = comparison.reset_index().rename(columns={"Horizon": "Kỳ dự báo", "StateOnly": "Chỉ dùng nhóm hiện tại", "MigrationAware": "Có thêm Migration", "MigrationGain": "Phần cải thiện của Migration"})
            display_gain[[c for c in ["Kỳ dự báo", "Chỉ dùng nhóm hiện tại", "Có thêm Migration", "Phần cải thiện của Migration"] if c in display_gain.columns]] *= 100
            st.markdown("**Migration có thêm thông tin hay không?**")
            st.dataframe(display_gain, use_container_width=True)
            st.caption("Nếu phần cải thiện của Migration dương và ổn định qua nhiều giai đoạn, đó mới là bằng chứng rằng việc chuyển nhóm có thể bổ sung thông tin ngoài trạng thái nhóm hiện tại. Một vài kỳ dương riêng lẻ chưa đủ để kết luận.")

    st.subheader("Chi tiết các lần chuyển nhóm")
    display_migration = migration.copy()
    rename_map = {"Ticker": "Mã cổ phiếu", "Date": "Ngày", "ClusterLabel": "Nhóm mới", "PreviousObservedCluster": "Nhóm trước", "Transition": "Thay đổi", "MigrationType": "Loại thay đổi", "AssignmentConfidence": "Độ chắc chắn", "MigrationSignal": "Có tín hiệu chuyển nhóm", "MigrationConfirmed": "Được xác nhận"}
    display_migration = display_migration.rename(columns={k: v for k, v in rename_map.items() if k in display_migration.columns})
    preferred = [c for c in ["Mã cổ phiếu", "Ngày", "Nhóm trước", "Nhóm mới", "Thay đổi", "Loại thay đổi", "Độ chắc chắn", "Có tín hiệu chuyển nhóm", "Được xác nhận"] if c in display_migration.columns]
    st.dataframe(display_migration[preferred], use_container_width=True)

    st.subheader("Thay đổi thành phần VN30")
    st.caption("Chỉ dùng để đánh dấu thời điểm rổ VN30 thay đổi. Số lượng vẫn là 30; điều quan trọng là mã nào vào và mã nào ra.")
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
