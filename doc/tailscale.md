# AnnoPilot qua Tailscale trên Windows

Tailscale đã được cài trên máy. Các bước đăng nhập và thay đổi quyền Tailscale cần thực hiện bằng tài khoản của bạn; dịch vụ Windows hiện yêu cầu quyền Administrator.

## Chia sẻ web để người xem chỉ cần trình duyệt

Dùng **Funnel** để người xem không cần cài Tailscale. Giữ backend và frontend đang chạy, mở **PowerShell → Run as administrator** rồi chạy:

```powershell
cd D:\AI\repo\Build_Phase\AnnoPilot-qminh
.\Start-Funnel.ps1
```

Script kiểm tra frontend/backend, rồi bật Funnel HTTPS cổng 443 trỏ tới frontend cổng 3000. Nếu Tailscale in đường dẫn yêu cầu bật Funnel hoặc HTTPS, mở đường dẫn đó bằng tài khoản của bạn, hoàn tất rồi chạy lại script. Gửi đúng link HTTPS trong kết quả `funnel status` cho người dùng. Với tên thiết bị/mạng hiện tại, địa chỉ dự kiến là `https://annopilot-t003.tailb9a83c.ts.net`; cần kết quả lệnh xác nhận Funnel đang bật trước khi chia sẻ ra ngoài.

Đây là link công khai. Danh sách email/PIN của Cloudflare không chuyển sang Funnel, và AnnoPilot hiện chưa có lớp đăng nhập riêng. Người có link có thể dùng các chức năng hiện có của web. Nếu muốn tiếp tục giới hạn hai email, cần bổ sung xác thực web thay vì dùng Funnel trực tiếp.

Funnel dùng `--bg`, nên đóng PowerShell không dừng chia sẻ. Máy chủ, Tailscale, backend và frontend vẫn cần hoạt động. Để dừng Funnel:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' funnel --https=443 off
```

Không cần chạy thêm Cloudflare Tunnel cho đường truy cập này. Các bước Serve bên dưới dành cho trường hợp muốn truy cập riêng qua ứng dụng Tailscale.

## Chọn cách truy cập

- **Serve:** link HTTPS ổn định, riêng cho người được cấp quyền trong mạng Tailscale. Máy chủ và máy người test đều cài, đăng nhập và bật Tailscale. Phù hợp để bắt đầu với hai người hiện tại.
- **Funnel:** link HTTPS công khai, người xem không cần cài Tailscale. Funnel không tự tạo màn hình đăng nhập hay danh sách email được phép như Cloudflare. Dùng script phía trên để chia sẻ web công khai; cần bổ sung lớp xác thực riêng nếu muốn giới hạn người dùng.

Tailscale Serve cấp địa chỉ dạng `https://annopilot-t003.<ten-tailnet>.ts.net`. Phần tailnet được lấy từ tài khoản, không phải tên miền riêng chỉ gồm `AnnoPilot_T003`. Địa chỉ giữ nguyên khi tên thiết bị và tên tailnet giữ nguyên. Máy chủ vẫn cần bật và chạy AnnoPilot.

## 1. Cài và đăng nhập

Tải bản Windows từ [Tailscale Downloads](https://tailscale.com/download/windows), chạy trình cài đặt, mở Tailscale và đăng nhập bằng tài khoản của bạn. Dùng gói Personal nếu dự án đáp ứng điều kiện sử dụng cá nhân, phi thương mại.

Mở PowerShell mới sau khi cài. Các lệnh bên dưới dùng đường dẫn cài mặc định để không phụ thuộc PATH:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' status
& 'C:\Program Files\Tailscale\tailscale.exe' set --hostname=annopilot-t003
```

Nếu Windows yêu cầu quyền quản trị cho thao tác cấu hình, chạy PowerShell bằng Run as administrator. Bật MagicDNS và HTTPS Certificates trong trang DNS của bảng quản trị Tailscale nếu chưa bật. Đây là thay đổi cấu hình riêng của mạng Tailscale, không thay tên máy Windows.

## 2. Chạy backend và frontend

Nếu backend đang chạy khỏe ở cổng 8000 thì giữ nguyên. Nếu chưa chạy:

```powershell
cd D:\AI\repo\Build_Phase\AnnoPilot-qminh\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Hostname hiện tại `annopilot-t003.tailb9a83c.ts.net` đã được cho phép sẵn trong `frontend/vite.config.ts`; với địa chỉ này chỉ cần khởi động frontend bằng `npm.cmd run dev` trong thư mục frontend. Không cần đặt lại biến môi trường sau mỗi lần mở terminal.

Nếu dùng hostname khác, trong terminal frontend, dừng Vite cũ bằng Ctrl+C nếu nó đang chạy. Sau đó chạy cả khối sau trong cùng cửa sổ PowerShell:

```powershell
cd D:\AI\repo\Build_Phase\AnnoPilot-qminh\frontend
$annoTsStatus = & 'C:\Program Files\Tailscale\tailscale.exe' status --json | ConvertFrom-Json
if ($annoTsStatus.BackendState -ne 'Running' -or -not $annoTsStatus.Self.DNSName) { throw 'Hay dang nhap va bat Tailscale truoc.' }
$env:ANNO_ALLOWED_HOSTS = $annoTsStatus.Self.DNSName.TrimEnd('.')
Write-Host "Hostname frontend chap nhan: $env:ANNO_ALLOWED_HOSTS"
npm.cmd run dev
```

Giữ terminal này mở. `ANNO_ALLOWED_HOSTS` cho phép đúng hostname Tailscale, tránh lỗi `Blocked request. This host ... is not allowed`. Giá trị chỉ áp dụng trong terminal hiện tại; khi khởi động frontend bằng cửa sổ mới, chạy lại khối lệnh trên. API vẫn đi qua frontend tại `/api`, không phải mở thêm cổng backend ra ngoài.

## 3. Bật Serve

Mở một PowerShell khác:

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' serve --bg http://127.0.0.1:3000
& 'C:\Program Files\Tailscale\tailscale.exe' serve status
```

Lần đầu có thể in đường dẫn yêu cầu bật HTTPS/Serve. Mở đường dẫn đó bằng tài khoản quản trị Tailscale, hoàn tất rồi chạy lại lệnh. Dùng đúng địa chỉ HTTPS được in trong kết quả, không tự đoán phần tên tailnet. Đây là link riêng, chưa phải link Funnel công khai.

## 4. Cấp quyền cho người test

Với mạng Tailscale mới chỉ dành cho dự án này, mở trang quản trị **Users → Invite external users**, mời người test với vai trò Member. Người kia nhận lời mời, cài Tailscale, đăng nhập bằng tài khoản riêng và chọn đúng mạng Tailscale của dự án. Quyền kết nối tuân theo các quy tắc truy cập của mạng; nếu mạng có quy tắc riêng, cần cho phép người đó tới máy AnnoPilot ở cổng HTTPS 443.

Nếu tài khoản đang quản lý nhiều máy khác, có thể dùng tính năng **Share** riêng máy AnnoPilot thay cho mời vào cả mạng. Chia sẻ máy cũng chịu các quy tắc truy cập Tailscale.

Người test bật Tailscale, mở link HTTPS Serve trong trình duyệt và kiểm tra: chọn bộ dữ liệu, mở ảnh, xem kết quả QC. Không cần chạy huấn luyện để thử kết nối. Tắt Cloudflare Tunnel sau khi xác nhận đường Tailscale hoạt động nếu muốn chỉ dùng đường truy cập mới.

## Kiểm tra và dừng

```powershell
& 'C:\Program Files\Tailscale\tailscale.exe' status
& 'C:\Program Files\Tailscale\tailscale.exe' serve status
& 'C:\Program Files\Tailscale\tailscale.exe' serve --https=443 off
```

Lệnh cuối dừng cấu hình Serve ở cổng 443; backend/frontend vẫn chạy. `--bg` giữ Serve chạy nền, nhưng không tự khởi động AnnoPilot khi bật máy.

Nguồn: [Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve), [Serve CLI](https://tailscale.com/docs/reference/tailscale-cli/serve), [mời người dùng](https://tailscale.com/docs/features/sharing/how-to/invite-any-user), [Funnel](https://tailscale.com/docs/features/tailscale-funnel), [Personal plan](https://tailscale.com/pricing).
