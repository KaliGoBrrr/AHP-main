import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import requests

# Hàm gọi API
def safe_api_request(method, endpoint, data=None):
    base_url = "http://127.0.0.1:8000"
    url = f"{base_url}/{endpoint}"
    try:
        if method.upper() == "POST":
            response = requests.post(url, json=data)
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
        html += f"<tr><td>{labels[i]}</td>" + "".join(f"<td>{val:.4f}</td>" for val in row) + "</tr>"
    html += "</table>"
    return html

# Hàm chuyển đổi giá trị thành số thực
def convert_to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if '/' in value:
            try:
                num, denom = map(float, value.split('/'))
                if denom == 0:
                    raise ValueError("Mẫu số không được bằng 0")
                return num / denom
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"Phân số không hợp lệ: {value}")
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Giá trị không phải số: {value}")
    raise ValueError(f"Giá trị không hợp lệ: {value}")

# Hàm tính trọng số và CR từ ma trận so sánh
def compute_weights_and_cr(matrix, n):
    col_sums = np.sum(matrix, axis=0)
    if np.any(col_sums == 0):
        raise ValueError("Ma trận chứa cột có tổng bằng 0")
    normalized_matrix = matrix / col_sums
    weights = np.mean(normalized_matrix, axis=1)
    weights = weights / np.sum(weights)  # Chuẩn hóa
    weighted_sum = np.dot(matrix, weights)
    consistency_vector = weighted_sum / weights
    lambda_max = np.mean(consistency_vector)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    ri_values = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_values.get(n, 1.5)
    cr = ci / ri if ri != 0 else 0
    return weights, cr

# Hàm chính để tính AHP từ Excel
def calculate_ahp_from_excel():
    st.title("Tính AHP từ file Excel")
    st.markdown("Tải lên file Excel chứa ma trận so sánh tiêu chí (sheet 'MTSS') và ma trận so sánh xe (sheets 'MTSS - {tên tiêu chí}').")

    uploaded_file = st.file_uploader("Chọn file Excel", type=["xlsx"])

    if uploaded_file:
        try:
            # Đọc file Excel
            xls = pd.ExcelFile(uploaded_file)
            sheets = xls.sheet_names

            # Kiểm tra sheet tiêu chí
            if "MTSS" not in sheets:
                st.error("File Excel phải chứa sheet 'MTSS' cho ma trận so sánh tiêu chí.")
                return

            # Đọc ma trận tiêu chí
            criteria_df = pd.read_excel(uploaded_file, sheet_name="MTSS", index_col=0)
            criteria_names = criteria_df.index.tolist()
            if len(criteria_names) < 2:
                st.error("Cần ít nhất 2 tiêu chí trong sheet 'MTSS'.")
                return

            # Chuyển đổi dữ liệu tiêu chí
            try:
                criteria_df = criteria_df.applymap(convert_to_float)
            except ValueError as e:
                st.error(f"Lỗi khi chuyển đổi dữ liệu ma trận tiêu chí: {str(e)}")
                return

            criteria_df = criteria_df.fillna(0)
            criteria_matrix = criteria_df.to_numpy(dtype=float)

            # Kiểm tra ma trận tiêu chí
            if criteria_matrix.shape != (len(criteria_names), len(criteria_names)):
                st.error("Ma trận tiêu chí phải là ma trận vuông.")
                return
            if np.any(np.isnan(criteria_matrix)) or np.any(np.isinf(criteria_matrix)):
                st.error("Ma trận tiêu chí chứa NaN hoặc giá trị vô cực.")
                return

            # Tính trọng số tiêu chí và CR
            criteria_weights, criteria_cr = compute_weights_and_cr(criteria_matrix, len(criteria_names))

            # Hiển thị ma trận và trọng số tiêu chí
            st.markdown("### Ma trận so sánh cặp tiêu chí")
            st.markdown(matrix_to_html(criteria_matrix, criteria_names), unsafe_allow_html=True)
            st.markdown(f"### Trọng số tiêu chí (CR = {criteria_cr:.4f} - {'Nhất quán' if criteria_cr < 0.1 else 'Không nhất quán'})")
            weights_html = "<ul>" + "".join(f"<li>{name}: {w:.4f}</li>" for name, w in zip(criteria_names, criteria_weights)) + "</ul>"
            st.markdown(weights_html, unsafe_allow_html=True)

            if criteria_cr >= 0.1:
                st.error(f"Ma trận tiêu chí không nhất quán (CR = {criteria_cr:.4f}). Vui lòng kiểm tra lại file Excel.")
                return

            # Đọc ma trận so sánh xe
            vehicle_names = None
            alternative_matrices = []
            consistency_ratios = []
            for criterion in criteria_names:
                sheet_name = f"MTSS - {criterion}"
                if sheet_name not in sheets:
                    st.error(f"File Excel phải chứa sheet '{sheet_name}' cho tiêu chí '{criterion}'.")
                    return
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, index_col=0)
                if vehicle_names is None:
                    vehicle_names = df.index.tolist()
                    if len(vehicle_names) < 2:
                        st.error("Cần ít nhất 2 xe trong ma trận so sánh.")
                        return
                elif df.index.tolist() != vehicle_names:
                    st.error(f"Sheet '{sheet_name}' có danh sách xe không khớp với các sheet khác.")
                    return

                # Chuyển đổi dữ liệu xe
                try:
                    df = df.applymap(convert_to_float)
                except ValueError as e:
                    st.error(f"Lỗi khi chuyển đổi dữ liệu ma trận '{criterion}': {str(e)}")
                    return

                df = df.fillna(0)
                matrix = df.to_numpy(dtype=float)

                # Kiểm tra ma trận xe
                if matrix.shape != (len(vehicle_names), len(vehicle_names)):
                    st.error(f"Ma trận cho tiêu chí '{criterion}' phải là ma trận vuông.")
                    return
                if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
                    st.error(f"Ma trận cho tiêu chí '{criterion}' chứa NaN hoặc giá trị vô cực.")
                    return

                # Tính trọng số xe và CR
                weights, cr = compute_weights_and_cr(matrix, len(vehicle_names))
                consistency_ratios.append(cr)
                alternative_matrices.append(matrix)

                # Hiển thị ma trận xe
                st.markdown(f"### Ma trận so sánh cặp xe ({criterion})")
                st.markdown(matrix_to_html(matrix, vehicle_names) + f"<p>CR = {cr:.4f} ({'Nhất quán' if cr < 0.1 else 'Không nhất quán'})</p>", unsafe_allow_html=True)

            # Tính điểm AHP
            n_vehicles = len(vehicle_names)
            scores = np.zeros(n_vehicles)
            for i, matrix in enumerate(alternative_matrices):
                weights, _ = compute_weights_and_cr(matrix, n_vehicles)
                scores += criteria_weights[i] * weights

            # Tạo kết quả xếp hạng
            ranking = [(vehicle_names[i], scores[i]) for i in range(n_vehicles)]
            ranking.sort(key=lambda x: x[1], reverse=True)

            # Hiển thị kết quả
            inconsistent_matrices = [i for i, cr in enumerate(consistency_ratios) if cr >= 0.1]
            warning = f"Cảnh báo: Ma trận không nhất quán (CR ≥ 0.1) cho tiêu chí: {[criteria_names[i] for i in inconsistent_matrices]}" if inconsistent_matrices else None

            html_result = "<h3>Kết quả xếp hạng:</h3>"
            if warning:
                html_result += f"<p style='color:red'>{warning}</p>"
            html_result += "<table border='1'><tr><th>Xe</th><th>Điểm AHP</th></tr>"
            for name, score in ranking:
                html_result += f"<tr><td>{name}</td><td>{score:.4f}</td></tr>"
            html_result += "</table>"
            st.markdown(html_result, unsafe_allow_html=True)

            # Biểu đồ cột
            fig = px.bar(x=[name for name, score in ranking], y=[score for name, score in ranking],
                         labels={'x': 'Xe', 'y': 'Điểm AHP'}, title='Xếp hạng xe')
            fig.update_traces(texttemplate='%{y:.4f}', textposition='outside')
            fig.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig)

            # Biểu đồ tròn
            weights_fig = px.pie(values=criteria_weights, names=criteria_names,
                                title='Trọng số tiêu chí', hole=0.4)
            weights_fig.update_traces(textinfo='percent+label')
            st.plotly_chart(weights_fig)

            # Lưu log
            log_data = {
                "weights": criteria_weights.tolist(),
                "top_result": [[name, score] for name, score in ranking[:3]],
                "criteria_matrices": [{"vehicle_" + str(i+1) + "_vs_" + str(j+1): matrix[i][j]
                                      for i in range(len(vehicle_names))
                                      for j in range(i + 1, len(vehicle_names))}
                                     for matrix in alternative_matrices]
            }
            response = safe_api_request("POST", "log_calculation", log_data)
            if "error" in response:
                st.error(f"Lỗi khi lưu log: {response['error']} (Mã trạng thái: {response.get('status_code', 'N/A')})")
            else:
                st.success("Lưu log tính toán thành công.")

        except Exception as e:
            st.error(f"Lỗi khi xử lý file Excel: {str(e)}")

# Chạy ứng dụng
if __name__ == "__main__":
    calculate_ahp_from_excel()