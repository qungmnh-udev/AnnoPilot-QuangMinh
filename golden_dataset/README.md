# CVAT labels cho Golden Dataset AnnoPilot

## Cách dùng

1. Tạo Project trên CVAT, vào phần Labels → Raw.
2. Mở `cvat-labels.json`, copy toàn bộ nội dung JSON và dán vào Raw.
3. Lưu labels, tạo Task thuộc Project đó và upload ảnh cần gán nhãn.
4. Nếu tạo Task độc lập, dán JSON vào Labels → Raw của Task.

Đây là cấu hình label, không phải file annotation để upload qua Import annotations.
Tất cả label dùng rectangle (BBox 2D). Không khai báo ID: CVAT tự cấp ID.
Nếu Task thuộc Project đã có labels, quản lý taxonomy ở Project.
`classes.txt` giữ thứ tự class hiện tại của training/dataset/data.yaml; ID nội bộ của CVAT không phải class ID YOLO.

## Taxonomy và quy ước

| YOLO ID | Label | Quy ước cho bộ Golden này |
| --- | --- | --- |
| 0 | pedestrian | Người đi bộ, đứng hoặc ngồi; người chưa ngồi/điều khiển xe thuộc nhóm này. |
| 1 | rider | Người đang ngồi/điều khiển xe đạp hoặc xe máy; hành khách trên xe hai bánh cũng thuộc rider. |
| 2 | car | Ô tô con, SUV, van chở người. |
| 3 | truck | Xe tải, pickup, van chở hàng nếu có đủ bằng chứng phân biệt. |
| 4 | bus | Xe buýt, xe khách, minibus có cấu trúc chở khách rõ ràng. |
| 5 | train | Mỗi toa hoặc đầu máy phân biệt được có một box; không vẽ một box bao cả đoàn tàu. |
| 6 | motorcycle | Xe máy; box xe riêng, không gộp người lái. |
| 7 | bicycle | Xe đạp; box xe riêng, không gộp người đi xe. |
| 8 | traffic light | Mỗi cụm đèn điều khiển giao thông có một box; không bao cột đèn. |
| 9 | traffic sign | Mỗi biển báo riêng có một box; không bao cột và không gán cho bảng quảng cáo. |

Các ranh giới trên là quy ước đề xuất cho dataset này, không phải xác nhận tiêu chuẩn BDD100K.
Nếu đề bài hoặc nguồn dữ liệu có hướng dẫn riêng, thống nhất lại quy ước trước khi annotate toàn bộ.
Không thêm class vehicle, person, background, missing_object hoặc wrong_class.

## Thuộc tính

- `occlusion_level`: unknown (chưa đánh giá), none (không che), partial (che một phần, nhận dạng rõ), heavy (che phần lớn nhưng vẫn đủ bằng chứng xác định vật thể).
- `truncation`: unknown (chưa đánh giá), none (nằm trọn trong ảnh), partial (bị cắt bởi mép ảnh).
- `label_certainty`: needs_review (cần kiểm chứng) hoặc clear (reviewer đã xác nhận class).
- `review_note`: ghi chú tự do; dùng cho trường hợp khó hoặc quyết định phân loại.

Dùng thêm cờ Occluded có sẵn của CVAT: bật khi occlusion_level là partial/heavy, tắt khi none.
Truncation do mép ảnh không tự động có nghĩa occluded.
Các thuộc tính mutable có thể thay đổi theo frame nếu dùng track.
Default unknown/needs_review tránh coi box mới vẽ là đã kiểm chứng.

## Quy tắc box và kiểm chứng Golden

- Annotate toàn bộ vật thể thuộc taxonomy trên toàn ảnh, kể cả vùng không có prediction/warning.
- Box ôm toàn bộ phạm vi vật thể có thể suy ra đáng tin cậy, rồi giới hạn trong kích thước ảnh. Không đoán phần không thấy nếu thiếu bằng chứng.
- Vật thể bị che vẫn annotate nếu nhận dạng và phạm vi đủ chắc chắn; trường hợp không chắc giữ needs_review và ghi chú.
- Người lái và xe là hai đối tượng riêng; có thể có box chồng nhau.
- Không tự loại vật thể nhỏ theo min_box_area của QC: đó là ngưỡng thuật toán, không phải định nghĩa ground truth.
- Reviewer độc lập kiểm tra toàn ảnh, giải quyết mọi needs_review và unknown trước khi chốt Golden.
- Nếu không thể giải quyết trường hợp mơ hồ, loại frame đó khỏi benchmark chính và ghi vào danh sách loại riêng. Evaluator hiện không có ignore region.
- Ảnh không có vật thể trong taxonomy vẫn được giữ và xác nhận là ảnh rỗng.
- Lưu annotation cần QC và Golden độc lập; không sửa bản cần QC thành Golden trước khi đo lỗi.

## Đưa dữ liệu về AnnoPilot

Export CVAT for images XML cùng ảnh để import annotation vào AnnoPilot.
Golden Benchmark dùng JSON riêng theo backend/app/schemas/evaluation.py:
verified_by, notes, includes_unwarned_regions=true, samples[{file_name, annotations[{label, geometry:{x1,y1,x2,y2}}]}].
Cấu hình labels này không tạo Golden JSON hoặc box thực tế.
Parser CVAT của AnnoPilot giữ attributes và occluded, nhưng evaluator Golden hiện chỉ dùng class và box.
label_certainty=needs_review không làm evaluator bỏ qua box.
Các metric chia theo occlusion/truncation cần xử lý bổ sung; app hiện không tự tính.

## Tài liệu CVAT

- https://docs.cvat.ai/docs/workspace/tasks-page/
- https://docs.cvat.ai/docs/api_sdk/sdk/reference/models/label/
- https://docs.cvat.ai/docs/api_sdk/sdk/reference/models/attribute/
