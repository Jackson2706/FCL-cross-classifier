import numpy as np

def metric_average_forgetting(accuracy_matrix):
    """
    Tính Average Forgetting dựa trên chênh lệch giữa Peak Accuracy (trong quá khứ)
    và Current Accuracy của TẤT CẢ các task.
    
    Args:
        accuracy_matrix (list of lists): Ma trận [Time_Steps x Num_Tasks].
                                         Dòng là các lần eval, Cột là các Task.
    
    Returns:
        float: Giá trị Average Forgetting.
    """
    # Nếu chưa có lịch sử hoặc mới chỉ có 1 dòng, chưa có gì để quên
    if not accuracy_matrix or len(accuracy_matrix) < 1:
        return 0.0

    # Lấy dòng accuracy mới nhất (trạng thái hiện tại)
    current_accuracies = accuracy_matrix[-1]
    num_tasks = len(current_accuracies)
    
    forgetting_sum = 0.0

    # Duyệt qua từng Task (từng cột)
    for j in range(num_tasks):
        # Lấy lịch sử accuracy của task j (toàn bộ cột j từ đầu đến giờ)
        # Lưu ý: Lấy max của cả quá khứ lẫn hiện tại để đảm bảo Forgetting >= 0
        history_task_j = [row[j] for row in accuracy_matrix]
        
        # Tìm độ chính xác cao nhất từng đạt được (Peak Performance)
        peak_acc = max(history_task_j)
        
        # Độ chính xác hiện tại
        current_acc = current_accuracies[j]
        
        # Tính forgetting cho task j
        # Nếu task chưa bao giờ được học, Peak ~ Current ~ Random Guess -> Forgetting ~ 0 (Hợp lý)
        forgetting_task_j = peak_acc - current_acc
        
        forgetting_sum += forgetting_task_j

    # Trung bình cộng trên tổng số task
    average_forgetting = forgetting_sum / num_tasks
    
    return average_forgetting