/** Display labels only. API values, route keys and taxonomy identifiers stay stable. */
const labels: Record<string, string> = {
  Home: 'Trang chủ', 'Model QC': 'Kiểm tra nhãn', Dataset: 'Bộ dữ liệu',
  Dashboard: 'Tổng quan', Settings: 'Cài đặt', Difficulty: 'Độ khó',
  'Smart Sampling': 'Chọn mẫu kiểm tra', Workload: 'Phân công kiểm tra',
  'Review Queue': 'Danh sách cần kiểm tra',
  EASY: 'Dễ', MEDIUM: 'Trung bình', HARD: 'Khó', UNAVAILABLE: 'Chưa có dữ liệu',
  HIGH: 'Cao', LOW: 'Thấp', CLEAN: 'Chưa có nghi vấn',
  PENDING: 'Chưa kiểm tra', REVIEWED: 'Đã kiểm tra', NO_ISSUE: 'Chưa thấy lỗi', ISSUE_FOUND: 'Có lỗi', SKIPPED: 'Đã bỏ qua',
  QUEUED: 'Đang chờ', RUNNING: 'Đang chạy', COMPLETED: 'Hoàn tất', FAILED: 'Không thành công',
  CANCELLED: 'Đã dừng', INTERRUPTED: 'Đã gián đoạn', STALE: 'Cần chạy lại',
  NOT_SUBMITTED: 'Chưa nộp', SUBMITTED: 'Đã nộp, chờ duyệt', RETURNED: 'Yêu cầu sửa lại', ACCEPTED: 'Đã chấp nhận',
  RESOLVED: 'Đã xử lý', FALSE_POSITIVE: 'Cảnh báo sai', OPEN: 'Chưa xử lý',
  BBOX_2D: 'Khung bao vật thể (2D)', SEGMENTATION: 'Vùng phân đoạn', KEYPOINT: 'Điểm đặc trưng / tư thế',
  CUBOID_3D: 'Khối hộp (3D)', AUTO: 'Tự nhận diện', MIXED: 'Nhiều dạng nhãn', UNKNOWN: 'Chưa có nhãn',
  POLYGON: 'Đa giác', MASK: 'Mặt nạ phân đoạn',
  bbox: 'Khung bao', polygon: 'Đa giác', mask: 'Mặt nạ', keypoint: 'Điểm đặc trưng', cuboid: 'Khối hộp',
  random: 'Ngẫu nhiên', difficulty: 'Độ khó', edge: 'Trường hợp hiếm', coverage_fallback: 'Bổ sung độ phủ',
  visual: 'Đặc điểm ảnh', annotation: 'Đặc điểm nhãn', overall: 'Tổng hợp',
  visual_weights: 'Trọng số đặc điểm ảnh', annotation_weights: 'Trọng số đặc điểm nhãn',
  overall_weights: 'Trọng số điểm tổng hợp', rare_class_threshold: 'Ngưỡng xác định loại vật thể hiếm',
  object_count: 'Số vật thể', class_diversity: 'Độ đa dạng loại vật thể', small_objects: 'Vật thể nhỏ',
  overlap: 'Mức chồng lấn', occlusion: 'Mức che khuất', truncation: 'Vật thể bị cắt',
  brightness: 'Độ sáng', contrast: 'Độ tương phản', blur: 'Độ mờ', edge_density: 'Mật độ đường nét',
  baseline: 'Một mô hình', multi_model: 'Nhiều mô hình',
};

export function displayLabel(value: string): string {
  return labels[value] ?? value.replaceAll('_', ' ').replaceAll('.', ' · ');
}

const messages: Record<string, string> = {
  'Failed to fetch': 'Không kết nối được máy chủ. Kiểm tra kết nối rồi thử lại.',
  'NetworkError when attempting to fetch resource.': 'Không kết nối được máy chủ. Kiểm tra kết nối rồi thử lại.',
  'Dataset not found': 'Không tìm thấy bộ dữ liệu. Hãy tải lại danh sách.',
  'Sample not found': 'Không tìm thấy ảnh. Hãy tải lại bộ dữ liệu.',
  'QC Issue not found': 'Nghi vấn không còn tồn tại. Hãy tải lại kết quả kiểm tra.',
  'Dataset name must contain 1–200 characters': 'Tên bộ dữ liệu phải có từ 1 đến 200 ký tự.',
  'Unsupported annotation task': 'Dạng nhãn này chưa được hỗ trợ.',
  'Media must be uploaded files': 'Vui lòng chọn tệp ảnh hoặc dữ liệu gốc để tải lên.',
  'Media unavailable': 'Không tìm thấy ảnh hoặc dữ liệu gốc.',
  'Sample does not belong to the specified dataset': 'Ảnh này không thuộc bộ dữ liệu đang chọn.',
  'Configure CVAT Task ID or Job ID': 'Mở cấu hình CVAT và nhập mã tác vụ hoặc mã công việc.',
  'CVAT Job ID does not belong to the configured Task ID': 'Mã công việc CVAT không thuộc tác vụ đã cấu hình. Hãy kiểm tra lại hai mã.',
  'Select an annotation job, not a ground-truth/consensus job': 'Hãy chọn công việc gán nhãn thông thường trên CVAT.',
  'CVAT task frame metadata is missing or invalid': 'Không đọc được thông tin ảnh trong tác vụ CVAT. Hãy kiểm tra lại tác vụ.',
  'Multiple local samples map to the same CVAT frame': 'Có nhiều mẫu trong AnnoPilot cùng trỏ đến một ảnh CVAT. Hãy kiểm tra dữ liệu nhập.',
  'Golden frames must be unique and belong to this dataset': 'Tên ảnh trong bộ nhãn chuẩn phải không trùng nhau và phải thuộc bộ dữ liệu đang chọn.',
  'Independent verifier is required': 'Hãy ghi tên người đã kiểm tra độc lập bộ nhãn chuẩn.',
  'Golden box exceeds original image dimensions': 'Khung bao trong bộ nhãn chuẩn vượt quá kích thước ảnh gốc.',
  'Benchmark does not belong to this dataset': 'Bộ nhãn chuẩn không thuộc bộ dữ liệu đang chọn.',
  'Benchmark frame is no longer available': 'Một ảnh trong bộ nhãn chuẩn không còn tồn tại. Hãy kiểm tra lại dữ liệu.',
  'Pilot sources share an identical checkpoint; do not count aliases as independent detectors': 'Các nguồn dự đoán dùng cùng một bản mô hình. Hãy chọn những mô hình độc lập để so sánh.',
  'Evaluation requires real independent detector predictions': 'Đánh giá cần dự đoán thật từ các mô hình độc lập; không dùng dữ liệu mô phỏng.',
  'Submit only a new dataset or a returned submission': 'Chỉ có thể nộp khi chưa nộp lần nào hoặc đã được yêu cầu sửa lại.',
  'Keep the same strategy across an acceptance cycle; use a separate dataset for the other strategy': 'Giữ nguyên cách kiểm tra trong một vòng nghiệm thu. Dùng bộ dữ liệu riêng nếu muốn đổi cách kiểm tra.',
  'Return/accept requires a submitted version with the same strategy': 'Cần nộp nghiệm thu theo đúng cách kiểm tra đã chọn trước khi trả lại hoặc chấp nhận.',
  'Provide a reason when rejecting an issue': 'Hãy ghi lý do bác bỏ nghi vấn.',
  'Reopen the issue before changing its review decision': 'Hãy hoàn tác kết quả xử lý trước khi đổi quyết định.',
};

/** Translate known service messages while retaining unknown technical diagnostics. */
export function displayMessage(message: string): string {
  if (messages[message]) return messages[message];
  const taxonomy = message.match(/CVAT taxonomy has no matching class for ['"](.+?)['"]/);
  if (taxonomy) return `CVAT chưa có loại vật thể tương ứng với “${taxonomy[1]}”. Hãy thêm hoặc chỉnh tên loại trong tác vụ CVAT cho khớp dữ liệu rồi gửi lại.`;
  const coverage = message.match(/^Missing inference\/import coverage for (.+) on (.+); empty results must be explicitly recorded$/);
  if (coverage) return `Chưa có dự đoán của ${coverage[1]} cho ảnh ${coverage[2]}. Hãy chạy nhận diện trước; ảnh không có vật thể cũng cần lưu kết quả rỗng.`;
  const failedFrame = message.match(/^Inference failed for (.+) on (.+)$/);
  if (failedFrame) return `Mô hình ${failedFrame[1]} chưa nhận diện được ảnh ${failedFrame[2]}. Hãy kiểm tra ảnh và chạy lại.`;
  const prefixes: Record<string, string> = {
    'CVAT Push failed: ': 'Không gửi được nhãn sang CVAT: ',
    'CVAT Sync failed: ': 'Không lấy được nhãn từ CVAT: ',
    'CVAT frame verification failed: ': 'Không xác minh được nhãn ảnh trên CVAT: ',
    'QC Audit failed: ': 'Không đối chiếu được nhãn: ',
    'Model detection failed: ': 'Không chạy được nhận diện: ',
    'Ensemble execution failed: ': 'Không hợp nhất được dự đoán: ',
    'Triad inference execution failed: ': 'Không hoàn tất được nhận diện bằng ba mô hình: ',
    'Failed to parse predictions JSON: ': 'Không đọc được tệp dự đoán JSON: ',
    'Import failed: ': 'Không nhập được dữ liệu: ',
  };
  for (const [prefix, translated] of Object.entries(prefixes)) {
    if (message.startsWith(prefix)) return translated + displayMessage(message.slice(prefix.length));
  }
  return message;
}
