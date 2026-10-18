import { displayMessage } from './uiText';

const tunnelMessage = 'Không hoàn tất được kết nối qua Cloudflare. Mở lại link gốc trong tab mới để kiểm tra phiên đăng nhập, rồi quay lại tải lại dữ liệu. Nếu vẫn lỗi, kiểm tra cửa sổ tunnel và kết nối máy chủ.';
const throughTunnel = () => typeof window !== 'undefined' && window.location.hostname.endsWith('.trycloudflare.com');

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch("/api" + path, options);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new Error(throughTunnel() ? tunnelMessage : "Không kết nối được máy chủ. Kiểm tra kết nối mạng và dịch vụ backend rồi thử lại.");
  }
  if (response.headers.get('content-type')?.includes('text/html')) {
    throw new Error(throughTunnel() ? tunnelMessage : 'API trả về trang HTML thay vì dữ liệu. Hãy kiểm tra kết nối frontend với backend và tải lại trang.');
  }
  if (!response.ok) {
    let message = `Không hoàn tất được yêu cầu (mã ${response.status}). Vui lòng thử lại.`;
    try {
      const body = await response.json();
      message =
        typeof body.detail === "string"
          ? body.detail
          : body.detail?.map((x: { msg: string }) => x.msg).join("; ") ||
            message;
    } catch {}
    throw new Error(displayMessage(message));
  }
  return response.json();
}
export const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
