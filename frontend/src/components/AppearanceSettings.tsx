import { Check, Moon, Sun, Monitor, Type } from "lucide-react";
import {
  useAppearance,
  type Theme,
  type FontSize,
} from "../appearance/AppearanceProvider";

export function AppearanceSettings() {
  const { theme, setTheme, fontSize, setFontSize } = useAppearance();
  return (
    <section
      className="card appearance-settings"
      aria-labelledby="appearance-heading"
    >
      <div className="card-title">
        <div>
          <h2 id="appearance-heading">Giao diện</h2>
          <p>Thay đổi có hiệu lực ngay và được lưu trên trình duyệt này.</p>
        </div>
        <Type size={22} />
      </div>
      <div className="appearance-grid">
        <fieldset>
          <legend>Chế độ màu</legend>
          <div className="preference-options">
            {(
              [
                ["light", "Sáng", Sun],
                ["dark", "Tối", Moon],
                ["system", "Theo thiết bị", Monitor],
              ] as const
            ).map(([value, label, Icon]) => (
              <label
                key={value}
                className={
                  theme === value ? "preference selected" : "preference"
                }
              >
                <input
                  type="radio"
                  name="appearance-theme"
                  value={value}
                  checked={theme === value}
                  onChange={() => setTheme(value as Theme)}
                />
                <Icon size={18} />
                <span>{label}</span>
                {theme === value && <Check size={16} aria-hidden="true" />}
              </label>
            ))}
          </div>
          <p className="muted">
            Chế độ Theo thiết bị tự đổi sáng / tối theo cài đặt của máy.
          </p>
        </fieldset>
        <fieldset>
          <legend>Cỡ chữ giao diện</legend>
          <div className="preference-options">
            {(["small", "medium", "large"] as const).map((value) => (
              <label
                key={value}
                className={
                  fontSize === value ? "preference selected" : "preference"
                }
              >
                <input
                  type="radio"
                  name="appearance-font-size"
                  value={value}
                  checked={fontSize === value}
                  onChange={() => setFontSize(value as FontSize)}
                />
                <span>{{ small: 'Nhỏ', medium: 'Vừa', large: 'Lớn' }[value]}</span>
                {fontSize === value && <Check size={16} aria-hidden="true" />}
              </label>
            ))}
          </div>
          <p className="muted">
            Đổi cỡ chữ trên toàn trang. Có thể cuộn bảng khi nội dung rộng hơn màn hình.
          </p>
        </fieldset>
      </div>
    </section>
  );
}
