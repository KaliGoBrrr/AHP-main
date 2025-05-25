import numpy as np

def calculate_ahp(criteria_weights, alternative_matrices, vehicle_names):
    """
    Tính toán tổng điểm AHP dựa trên trọng số tiêu chí và ma trận so sánh cặp thay thế.
    
    :param criteria_weights: Trọng số của các tiêu chí (list)
    :param alternative_matrices: Danh sách ma trận so sánh cặp cho từng tiêu chí (list of numpy arrays)
    :param vehicle_names: Danh sách tên xe (list)
    :return: Danh sách xe kèm điểm số, đã sắp xếp theo điểm giảm dần
    """
    # Kiểm tra đầu vào
    if not vehicle_names or not criteria_weights or not alternative_matrices:
        raise ValueError("Danh sách xe, trọng số tiêu chí hoặc ma trận không được rỗng.")
    n_vehicles = len(vehicle_names)
    n_criteria = len(criteria_weights)
    if len(alternative_matrices) != n_criteria:
        raise ValueError(f"Số ma trận so sánh cặp ({len(alternative_matrices)}) không khớp với số tiêu chí ({n_criteria}).")
    if abs(sum(criteria_weights) - 1.0) > 1e-6:
        raise ValueError("Tổng trọng số tiêu chí phải bằng 1.")
    if len(set(vehicle_names)) != len(vehicle_names):
        raise ValueError("Danh sách tên xe chứa giá trị trùng lặp.")

    # Kiểm tra ma trận
    for matrix in alternative_matrices:
        if matrix.shape != (n_vehicles, n_vehicles):
            raise ValueError(f"Ma trận so sánh cặp có kích thước không hợp lệ: {matrix.shape}")
        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            raise ValueError("Ma trận so sánh cặp chứa giá trị NaN hoặc vô cực.")

    # Tính trọng số cho từng xe theo từng tiêu chí
    alternative_weights = []
    for matrix in alternative_matrices:
        col_sums = np.sum(matrix, axis=0)
        if np.any(col_sums == 0):
            raise ValueError("Ma trận so sánh cặp chứa cột có tổng bằng 0.")
        normalized_matrix = matrix / col_sums
        weights = np.mean(normalized_matrix, axis=1)
        weights = weights / np.sum(weights)  # Chuẩn hóa
        alternative_weights.append(weights)

    # Tính điểm tổng
    scores = np.zeros(n_vehicles)
    for i in range(n_vehicles):
        for j in range(n_criteria):
            scores[i] += criteria_weights[j] * alternative_weights[j][i]

    # Tạo kết quả
    result = [(vehicle_names[i], round(scores[i], 4)) for i in range(n_vehicles)]
    return sorted(result, key=lambda x: x[1], reverse=True)