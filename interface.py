import streamlit as st
import numpy as np
import plotly.express as px
import requests
from datetime import datetime
import pytz
from ahp_calculator import calculate_ahp

# Hàm gọi API
def safe_api_request(method, endpoint, data=None):
    base_url = "http://127.0.0.1:8000"
    url = f"{base_url}/{endpoint}"
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            print(f"Sending POST to {url} with data: {data}")  # Debug dữ liệu gửi
            response = requests.post(url, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url)
        else:
            return {"error": "Phương thức HTTP không hỗ trợ"}
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "status_code": getattr(e.response, "status_code", None)}

# Hàm hiển thị ma trận dưới dạng HTML
def matrix_to_html(matrix, labels):
    html = "<table border='1'>"
    html += "<tr><th></th>" + "".join(f"<th>{label}</th>" for label in labels) + "</tr>"
    for i, row in enumerate(matrix):
        html += f"<tr><td>{labels[i]}</td>" + "".join(f"<td>{val:.2f}</td>" for val in row) + "</tr>"
    html += "</table>"
    return html

# Hàm kiểm tra trạng thái
def check_status():
    response = safe_api_request("GET", "check_status")
    if "error" in response:
        return {"criteria_count": 0, "vehicles_count": 0, "weights_saved": False}
    return response

# Bước 1: Quản lý tiêu chí và xe
def criteria_management_step():
    st.markdown("## Bước 1: Quản lý tiêu chí và xe")
    st.markdown("Thêm ít nhất 2 tiêu chí và 2 xe để tiếp tục.")

    # Lấy danh sách tiêu chí và xe
    def fetch_criteria():
        response = safe_api_request("GET", "get_criteria")
        return response.get("criteria", []) if "error" not in response else []

    def fetch_vehicles():
        response = safe_api_request("GET", "get_vehicles")
        return response.get("vehicles", []) if "error" not in response else []

    col1, col2 = st.columns(2)

    # Quản lý tiêu chí
    with col1:
        st.markdown("### Quản lý tiêu chí")
        new_criterion = st.text_input("Tên tiêu chí mới")
        if st.button("Thêm tiêu chí"):
            if new_criterion:
                response = safe_api_request("POST", "add_criterion", {"name": new_criterion})
                if "error" in response:
                    st.error(f"Lỗi: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})")
                else:
                    st.success(response.get("message", "Thêm tiêu chí thành công."))
                    st.rerun()
            else:
                st.error("Vui lòng nhập tên tiêu chí.")

        criteria = fetch_criteria()
        criterion_to_delete = st.selectbox("Chọn tiêu chí để xóa", [c["name"] for c in criteria], index=None)
        if st.button("Xóa tiêu chí"):
            if criterion_to_delete:
                response = safe_api_request("DELETE", f"delete_criterion/{criterion_to_delete}")
                if "error" in response:
                    st.error(f"Lỗi: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})")
                else:
                    st.success(response.get("message", "Xóa tiêu chí thành công."))
                    st.rerun()
            else:
                st.error("Vui lòng chọn tiêu chí để xóa.")

        st.markdown("#### Danh sách tiêu chí:")
        for c in criteria:
            st.write(f"- {c['name']} (index: {c['index']})")

    # Quản lý xe
    with col2:
        st.markdown("### Quản lý xe")
        vehicle_name = st.text_input("Tên xe")
        vehicle_image = st.text_input("URL hình ảnh", value="")
        vehicle_brand = st.text_input("Hãng xe", value="N/A")
        vehicle_year = st.number_input("Năm sản xuất", min_value=1900, max_value=2025, value=2020)
        vehicle_type = st.text_input("Loại xe", value="N/A")
        vehicle_engine = st.text_input("Động cơ", value="N/A")
        vehicle_price = st.text_input("Khoảng giá", value="N/A")
        if st.button("Thêm xe"):
            if vehicle_name:
                vehicle_data = {
                    "name": vehicle_name,
                    "image": vehicle_image,
                    "details": {
                        "brand": vehicle_brand,
                        "year": float(vehicle_year),
                        "type": vehicle_type,
                        "engine": vehicle_engine,
                        "price_range": vehicle_price
                    }
                }
                response = safe_api_request("POST", "add_vehicle", vehicle_data)
                if "error" in response:
                    st.error(f"Lỗi: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})")
                else:
                    st.success(response.get("message", "Thêm xe thành công."))
                    st.rerun()
            else:
                st.error("Vui lòng nhập tên xe.")

        vehicles = fetch_vehicles()
        vehicle_to_delete = st.selectbox("Chọn xe để xóa", [v["name"] for v in vehicles], index=None)
        if st.button("Xóa xe"):
            if vehicle_to_delete:
                response = safe_api_request("DELETE", f"delete_vehicle/{vehicle_to_delete}")
                if "error" in response:
                    st.error(f"Lỗi: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})")
                else:
                    st.success(response.get("message", "Xóa xe thành công."))
                    st.rerun()
            else:
                st.error("Vui lòng chọn xe để xóa.")

        st.markdown("#### Danh sách xe:")
        for v in vehicles:
            details = v.get("details", {})
            st.write(f"- {v['name']} ({details.get('brand', 'N/A')}, {details.get('year', 'N/A')})")

    # Kiểm tra trạng thái
    status = check_status()
    criteria_count = status.get("criteria_count", 0)
    vehicles_count = status.get("vehicles_count", 0)
    if criteria_count < 2 or vehicles_count < 2:
        st.error(f"Cần ít nhất 2 tiêu chí (hiện có: {criteria_count}) và 2 xe (hiện có: {vehicles_count}).")
    else:
        st.success("Đủ tiêu chí và xe. Nhấn 'Tiếp tục' để sang bước tiếp theo.")
        if st.button("Tiếp tục đến bước So sánh cặp tiêu chí"):
            st.session_state.step = "pairwise_comparison"
            st.rerun()

# Bước 2: So sánh cặp tiêu chí
def pairwise_comparison_step():
    st.markdown("## Bước 2: So sánh cặp các tiêu chí")
    st.markdown("""
    ### Thang đánh giá từ 1 đến 9:
    - **1**: Hai tiêu chí có tầm quan trọng ngang nhau
    - **3**: Tiêu chí A quan trọng hơn tiêu chí B một chút
    - **5**: Tiêu chí A quan trọng hơn tiêu chí B nhiều
    - **7**: Tiêu chí A quan trọng hơn tiêu chí B rất nhiều
    - **9**: Tiêu chí A quan trọng hơn tiêu chí B cực kỳ nhiều
    - **2, 4, 6, 8**: Giá trị trung gian
    """)

    def fetch_criteria():
        response = safe_api_request("GET", "get_criteria")
        return response.get("criteria", []) if "error" not in response else []

    criteria = fetch_criteria()
    criteria_names = [c["name"] for c in criteria]
    if len(criteria_names) < 2:
        st.error("Cần ít nhất 2 tiêu chí để so sánh. Vui lòng quay lại bước 1.")
        if st.button("Quay lại bước 1"):
            st.session_state.step = "criteria_management"
            st.rerun()
        return

    # Tạo slider cho so sánh cặp
    inputs = []
    for i in range(len(criteria_names)):
        for j in range(i + 1, len(criteria_names)):
            key = f"criteria_{criteria_names[i]}_vs_{criteria_names[j]}"
            value = st.session_state.get(key, 1.0)
            slider = st.slider(
                f"{criteria_names[i]} so với {criteria_names[j]}",
                min_value=0.1, max_value=9.0, step=0.1, value=value,
                key=key
            )
            inputs.append(slider)

    # Cập nhật ma trận và trọng số
    def update_matrix_and_weights(inputs):
        n = len(criteria_names)
        matrix = np.ones((n, n))
        input_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = inputs[input_idx]
                matrix[j][i] = 1.0 / inputs[input_idx]
                input_idx += 1

        col_sums = np.sum(matrix, axis=0)
        normalized_matrix = matrix / col_sums
        weights = np.mean(normalized_matrix, axis=1)
        weighted_sum = np.dot(matrix, weights)
        consistency_vector = weighted_sum / weights
        lambda_max = np.mean(consistency_vector)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_values.get(n, 1.5)
        cr = ci / ri if ri != 0 else 0

        matrix_html = matrix_to_html(matrix, criteria_names)
        weights_html = "<h4>Trọng số:</h4><ul>"
        for i, w in enumerate(weights):
            weights_html += f"<li>{criteria_names[i]}: {w:.4f}</li>"
        weights_html += f"</ul><p>CR = {cr:.4f} ({'Nhất quán' if cr < 0.1 else 'Không nhất quán'})</p>"

        return matrix_html, weights_html, weights, cr

    matrix_html, weights_html, weights, cr = update_matrix_and_weights(inputs)
    st.markdown("### Ma trận so sánh cặp")
    st.markdown(matrix_html, unsafe_allow_html=True)
    st.markdown("### Trọng số tiêu chí")
    st.markdown(weights_html, unsafe_allow_html=True)

    # Lưu trọng số
    if st.button("Lưu trọng số tiêu chí"):
        if cr >= 0.1:
            st.error(f"Ma trận không nhất quán (CR = {cr:.4f}). Vui lòng điều chỉnh giá trị so sánh.")
        else:
            response = safe_api_request("POST", "save_criteria_weights", {"weights": weights.tolist()})
            if "error" in response:
                st.error(f"Lỗi: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})")
            else:
                st.success(response.get("message", "Lưu trọng số thành công."))
                st.session_state.weights_saved = True
                st.rerun()

    # Tiếp tục
    if st.session_state.get("weights_saved", False):
        if st.button("Tiếp tục đến bước So sánh cặp xe"):
            st.session_state.step = "alternatives_comparison"
            st.rerun()
    else:
        st.error("Vui lòng lưu trọng số trước khi tiếp tục.")

    if st.button("Quay lại bước 1"):
        st.session_state.step = "criteria_management"
        st.rerun()

# Bước 3: So sánh cặp xe
def alternatives_comparison_step():
    st.markdown("## Bước 3: So sánh cặp các xe theo từng tiêu chí")
    st.markdown("Nhập giá trị so sánh cặp (1-9) cho tất cả xe theo từng tiêu chí.")

    def fetch_criteria():
        response = safe_api_request("GET", "get_criteria")
        return response.get("criteria", []) if "error" not in response else []

    def fetch_vehicles():
        response = safe_api_request("GET", "get_vehicles")
        return response.get("vehicles", []) if "error" not in response else []

    vehicles = fetch_vehicles()
    vehicle_names = [v["name"] for v in vehicles]
    criteria = fetch_criteria()
    criteria_names = [c["name"] for c in criteria]

    if len(criteria_names) == 0:
        st.error("Không có tiêu chí nào. Vui lòng quay lại bước 1.")
        if st.button("Quay lại bước 1"):
            st.session_state.step = "criteria_management"
            st.rerun()
        return
    if len(vehicle_names) < 2:
        st.error("Cần ít nhất 2 xe. Vui lòng quay lại bước 1.")
        if st.button("Quay lại bước 1"):
            st.session_state.step = "criteria_management"
            st.rerun()
        return

    # Tạo slider và ma trận cho mỗi tiêu chí
    alternative_inputs = []
    matrix_outputs = []
    for crit_idx, criterion in enumerate(criteria_names):
        st.markdown(f"### Tiêu chí: {criterion}")
        inputs = []
        for i in range(len(vehicle_names)):
            for j in range(i + 1, len(vehicle_names)):
                key = f"{criterion}_{vehicle_names[i]}_vs_{vehicle_names[j]}"
                value = st.session_state.get(key, 1.0)
                slider = st.slider(
                    f"{vehicle_names[i]} so với {vehicle_names[j]}",
                    min_value=0.1, max_value=9.0, step=0.1, value=value,
                    key=key
                )
                inputs.append(slider)
        alternative_inputs.append(inputs)

        # Tạo ma trận cho tiêu chí hiện tại
        matrix = np.ones((len(vehicle_names), len(vehicle_names)))
        input_idx = 0
        for i in range(len(vehicle_names)):
            for j in range(i + 1, len(vehicle_names)):
                matrix[i][j] = inputs[input_idx]
                matrix[j][i] = 1.0 / inputs[input_idx]
                input_idx += 1
        col_sums = np.sum(matrix, axis=0)
        normalized_matrix = matrix / col_sums
        weights = np.mean(normalized_matrix, axis=1)
        weighted_sum = np.dot(matrix, weights)
        consistency_vector = weighted_sum / weights
        lambda_max = np.mean(consistency_vector)
        ci = (lambda_max - len(vehicle_names)) / (len(vehicle_names) - 1) if len(vehicle_names) > 1 else 0
        ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_values.get(len(vehicle_names), 1.5)
        cr = ci / ri if ri != 0 else 0
        html = matrix_to_html(matrix, vehicle_names)
        html += f"<p>CR = {cr:.4f} ({'Nhất quán' if cr < 0.1 else 'Không nhất quán'})</p>"
        matrix_outputs.append(html)

        st.markdown(f"#### Ma trận so sánh cặp ({criterion})")
        st.markdown(html, unsafe_allow_html=True)

    # Tính toán AHP
    if st.button("Tính toán AHP"):
        response = safe_api_request("GET", "get_criteria_weights")
        if "error" in response:
            st.error(f"Lỗi: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})")
            return
        criteria_weights = response.get("weights", [1/len(criteria_names)] * len(criteria_names))

        alternative_matrices = []
        consistency_ratios = []
        ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        for crit_idx in range(len(criteria_names)):
            matrix = np.ones((len(vehicle_names), len(vehicle_names)))
            inputs = alternative_inputs[crit_idx]
            input_idx = 0
            for i in range(len(vehicle_names)):
                for j in range(i + 1, len(vehicle_names)):
                    matrix[i][j] = inputs[input_idx]
                    matrix[j][i] = 1.0 / inputs[input_idx]
                    input_idx += 1
            col_sums = np.sum(matrix, axis=0)
            normalized_matrix = matrix / col_sums
            weights = np.mean(normalized_matrix, axis=1)
            weighted_sum = np.dot(matrix, weights)
            consistency_vector = weighted_sum / weights
            lambda_max = np.mean(consistency_vector)
            ci = (lambda_max - len(vehicle_names)) / (len(vehicle_names) - 1) if len(vehicle_names) > 1 else 0
            ri = ri_values.get(len(vehicle_names), 1.5)
            cr = ci / ri if ri != 0 else 0
            consistency_ratios.append(cr)
            alternative_matrices.append(matrix)

        try:
            ranking = calculate_ahp(criteria_weights, alternative_matrices, vehicle_names)
        except Exception as e:
            st.error(f"Lỗi khi tính AHP: {str(e)}")
            return

        inconsistent_matrices = [i for i, cr in enumerate(consistency_ratios) if cr >= 0.1]
        warning = f"Cảnh báo: Ma trận không nhất quán (CR ≥ 0.1) cho tiêu chí: {[criteria_names[i] for i in inconsistent_matrices]}" if inconsistent_matrices else None

        html_result = "<h3>Kết quả xếp hạng:</h3>"
        if warning:
            html_result += f"<p style='color:red'>{warning}</p>"
        html_result += "<table border='1'><tr><th>Xe</th><th>Điểm AHP</th></tr>"
        for name, score in ranking:
            html_result += f"<tr><td>{name}</td><td>{score:.4f}</td></tr>"
        html_result += "</table>"
        html_result += "<h3>Độ nhất quán:</h3><ul>"
        for i, cr in enumerate(consistency_ratios):
            html_result += f"<li>Tiêu chí {criteria_names[i]}: CR = {cr:.4f} ({'Nhất quán' if cr < 0.1 else 'Không nhất quán'})</li>"
        html_result += "</ul>"

        st.markdown(html_result, unsafe_allow_html=True)

        fig = px.bar(x=[name for name, score in ranking], y=[score for name, score in ranking],
                     labels={'x': 'Xe', 'y': 'Điểm AHP'}, title='Xếp hạng xe')
        fig.update_traces(texttemplate='%{y:.4f}', textposition='outside')
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig)

        weights_fig = px.pie(values=criteria_weights, names=criteria_names,
                            title='Trọng số tiêu chí', hole=0.4)
        weights_fig.update_traces(textinfo='percent+label')
        st.plotly_chart(weights_fig)

        log_data = {
            "weights": criteria_weights,
            "top_result": [[name, score] for name, score in ranking[:3]],
            "criteria_matrices": [{"vehicle_" + str(i+1) + "_vs_" + str(j+1): matrix[i][j]
                                  for i in range(len(vehicle_names))
                                  for j in range(i + 1, len(vehicle_names))}
                                 for matrix in alternative_matrices]
        }
        response = safe_api_request("POST", "log_calculation", log_data)
        if "error" in response:
            st.error(f"Lỗi khi lưu log: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})")

    if st.button("Quay lại bước 2"):
        st.session_state.step = "pairwise_comparison"
        st.rerun()

# Bước 4: Xem lịch sử tính toán
def log_step():
    # Lưu bước trước đó để quay lại
    if "previous_step" not in st.session_state:
        st.session_state.previous_step = st.session_state.get("step", "criteria_management")

    st.markdown("## Xem lịch sử tính toán")

    def fetch_logs():
        response = safe_api_request("GET", "logs")
        if "error" in response:
            return f"<p style='color:red'>Lỗi: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})</p>", []
        logs = response
        if not logs:
            return "<p>Không có log nào.</p>", []
        html = "<h4>Lịch sử tính toán:</h4>"
        choices = []
        for log in logs:
            log_id = str(log.get("_id", ""))
            if not log_id or len(log_id) != 24:
                continue
            timestamp = log.get("timestamp", "N/A")
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                formatted_timestamp = dt.astimezone(pytz.timezone("Asia/Ho_Chi_Minh")).strftime("%d/%m/%Y %H:%M:%S")
            except (ValueError, TypeError):
                formatted_timestamp = timestamp
            weights = log.get("weights", [])
            top_result = log.get("top_result", [])
            html += f"<p><b>Thời gian:</b> {formatted_timestamp}</p>"
            html += "<p><b>Trọng số tiêu chí:</b> " + ", ".join([f"{w:.4f}" for w in weights]) + "</p>"
            html += "<p><b>Kết quả top:</b></p><ul>"
            for name, score in top_result:
                html += f"<li>{name}: {score:.4f}</li>"
            html += "</ul><hr>"
            choices.append((formatted_timestamp, log_id))
        return html, choices

    logs_html, choices = fetch_logs()
    st.markdown(logs_html, unsafe_allow_html=True)

    log_to_delete = st.selectbox("Chọn log để xóa (thời gian)", [c[0] for c in choices], index=None)
    if st.button("Xóa log"):
        if log_to_delete:
            log_id = next(c[1] for c in choices if c[0] == log_to_delete)
            response = safe_api_request("DELETE", f"logs/{log_id}")
            if "error" in response:
                st.error(f"Lỗi: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})")
            else:
                st.success(response.get("message", "Xóa log thành công."))
                st.rerun()
        else:
            st.error("Vui lòng chọn log để xóa.")

    if st.button("Quay lại bước trước"):
        st.session_state.step = st.session_state.previous_step
        st.rerun()

# Giao diện chính
st.title("Ứng dụng AHP - Đánh giá và xếp hạng xe")
st.markdown("Vui lòng thực hiện các bước theo thứ tự.")

# Thanh bên để truy cập lịch sử
with st.sidebar:
    st.header("Điều hướng")
    if st.button("Xem lịch sử tính toán"):
        st.session_state.previous_step = st.session_state.get("step", "criteria_management")
        st.session_state.step = "log"
        st.rerun()

# Khởi tạo trạng thái
if "step" not in st.session_state:
    st.session_state.step = "criteria_management"
if "weights_saved" not in st.session_state:
    st.session_state.weights_saved = False

# Hiển thị bước hiện tại
if st.session_state.step == "criteria_management":
    criteria_management_step()
elif st.session_state.step == "pairwise_comparison":
    pairwise_comparison_step()
elif st.session_state.step == "alternatives_comparison":
    alternatives_comparison_step()
elif st.session_state.step == "log":
    log_step()