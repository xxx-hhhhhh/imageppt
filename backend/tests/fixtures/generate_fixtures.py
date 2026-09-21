from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NAMES = ["01_simple_text", "02_business_slide", "03_technology", "04_academic", "05_infographic", "06_flowchart", "07_cards", "08_table", "09_chart", "10_poster", "11_dark_theme", "12_light_theme"]


def _font(size: int) -> ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, name in enumerate(NAMES):
        dark = name == "11_dark_theme"
        background = (21, 31, 52) if dark else (248, 250, 252)
        foreground = (242, 246, 250) if dark else (27, 47, 65)
        image = Image.new("RGB", (960, 540), background)
        draw = ImageDraw.Draw(image)
        draw.text((44, 30), name.replace("_", " ").title(), fill=foreground, font=_font(30))
        if name in {"01_simple_text", "04_academic", "10_poster"}:
            draw.multiline_text((70, 150), "Editable title\nSupporting paragraph", fill=foreground, font=_font(26), spacing=12)
        elif name in {"02_business_slide", "07_cards", "12_light_theme"}:
            for card in range(3 if name != "07_cards" else 5):
                x = 50 + card * 175
                draw.rounded_rectangle((x, 150, x + 145, 360), radius=14, fill=(220, 232, 244), outline=(84, 112, 145), width=2)
                draw.ellipse((x + 45, 175, x + 100, 230), fill=(42, 117, 185))
                draw.text((x + 20, 250), f"Card {card + 1}", fill=(27, 40, 55), font=_font(18))
        elif name == "03_technology":
            draw.line((80, 280, 880, 280), fill=(70, 170, 240), width=4)
            for x in (160, 400, 640):
                draw.ellipse((x - 35, 245, x + 35, 315), fill=(35, 145, 205))
        elif name == "05_infographic":
            for i in range(4):
                draw.rectangle((100 + i * 190, 350 - i * 45, 220 + i * 190, 350), fill=(80 + i * 30, 120, 205))
        elif name == "06_flowchart":
            for i in range(4):
                x = 70 + i * 215
                draw.rounded_rectangle((x, 230, x + 150, 310), radius=12, fill=(241, 179, 79), outline=(130, 88, 30), width=2)
                if i < 3:
                    draw.line((x + 150, 270, x + 215, 270), fill=foreground, width=3)
        elif name == "08_table":
            for row in range(4):
                for col in range(4):
                    draw.rectangle((100 + col * 170, 150 + row * 65, 270 + col * 170, 215 + row * 65), outline=(120, 140, 160), width=2)
        elif name == "09_chart":
            draw.line((90, 390, 90, 150), fill=foreground, width=2)
            draw.line((90, 390, 850, 390), fill=foreground, width=2)
            points = [(100, 350), (260, 290), (420, 320), (580, 210), (740, 180), (840, 230)]
            draw.line(points, fill=(218, 84, 84), width=5)
        paths.append(output_dir / f"{name}.png")
        image.save(paths[-1])
    return paths


if __name__ == "__main__":
    import sys
    generate(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/generated"))
