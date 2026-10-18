# Test link AnnoPilot bằng Cloudflare Quick Tunnel

Frontend/backend phải chạy trước. Từ gốc project, chạy `./Start-Tunnel.ps1` để dùng hai email mặc định đã cấu hình. Muốn dùng danh sách khác, truyền đầy đủ các email được phép:

```powershell
.\Start-Tunnel.ps1 -AllowedEmails "you@example.com","reviewer@example.com"
```

Script kiểm tra trang local và `/api/health`, tải connector Windows x64 từ release chính thức nếu chưa có, kiểm tra hỗ trợ `--allowed-mail`, rồi chạy Tunnel tại cổng 3000. Đổi cổng bằng `-Port 3001` nếu frontend đang ở cổng khác.

URL HTTPS `*.trycloudflare.com` được in trong terminal. Người truy cập cần xác minh email được cho phép bằng PIN. Giữ cửa sổ mở; Ctrl+C dừng link ngoài mà không tắt frontend/backend. Hostname thay đổi sau mỗi lần tạo Tunnel mới. Không cần gửi token CVAT hoặc mật khẩu cho người vận hành script.

Script đặt origin Host là `localhost:<port>` để Vite nhận request; không tắt host validation hoặc cho phép mọi hostname. Phiên bản connector không có bảo vệ email sẽ dừng và yêu cầu cập nhật hoặc dùng named Tunnel + Cloudflare Access, không tự mở ứng dụng không có bảo vệ.

Đây là luồng test ngắn hạn trên frontend hiện có. Hosting lâu dài nên dùng frontend build + reverse proxy và hostname cố định có Access. Chỉ bật Tunnel khi cần test, vì các email được phép có thể sử dụng toàn bộ thao tác của ứng dụng, chưa có phân quyền riêng theo người dùng.

Kiểm tra từ điện thoại dùng 4G/5G: đăng nhập email → mở trang → chọn dataset → xem ảnh → kiểm tra trạng thái QC. Không cần chạy training để chứng minh link hoạt động. Tránh upload ZIP lớn; Cloudflare giới hạn kích thước request theo gói. Triad inference dài hiện vẫn có thể timeout; training đã dùng job nền.

Nếu tải connector lỗi, kiểm tra truy cập GitHub trong PowerShell của bạn rồi chạy lại. Môi trường agent ngày 18/10 không kết nối được GitHub để tải connector nên chưa tạo/xác minh link ngoài. Trang local và API health đã trả HTTP 200.

Tài liệu: [Quick Tunnel và email protection](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/), [origin Host header](https://developers.cloudflare.com/tunnel/reference/origin-parameters/).

## Nhập PIN xong bị Forbidden, rồi Bad Request

Đọc log trong cửa sổ tunnel để phân biệt nguyên nhân. Nếu thấy:

```text
HTTP request authorization failed before origin selection
token issued in the future (iat)
```

Token bị từ chối ở bước xác thực do thời gian phát hành nằm trong tương lai so với đồng hồ máy chạy connector. Yêu cầu chưa tới Vite hoặc backend. Kiểm tra giờ Windows và đồng bộ lại; không cần bỏ bảo vệ email hay tắt kiểm tra Host của Vite.

Nếu Windows Time bị Disabled, mở **PowerShell → Run as administrator** và chạy:

```powershell
Set-Service -Name W32Time -StartupType Automatic
Start-Service -Name W32Time
w32tm /resync /rediscover
w32tm /query /status
```

Lệnh đồng bộ phải báo thành công. Có thể kiểm tra thêm tại Settings → Time & language → Date & time, bật Set time automatically và chọn Sync now. Nếu đồng bộ báo lỗi, xử lý lỗi đó trước khi thử đăng nhập lại.

Sau khi giờ đã đồng bộ, Ctrl+C dừng tunnel rồi chạy lại Start-Tunnel.ps1 với email của bạn. Mở link vừa tạo bằng cửa sổ ẩn danh mới và yêu cầu mã PIN mới. Không tải lại đường dẫn xác thực cũ. Dòng `request contains 0 authentication-state cookies, expected one` cho biết yêu cầu thiếu cookie trạng thái đăng nhập; bắt đầu lại từ link gốc trong cùng một trình duyệt, không chuyển giữa các cửa sổ hoặc trình duyệt khi nhập mã. Nếu vẫn còn lỗi cookie, kiểm tra cài đặt chặn cookie của trình duyệt.

Tham khảo: [Windows Time và w32tm](https://learn.microsoft.com/en-us/windows-server/networking/windows-time-service/windows-time-service-tools-and-settings).
