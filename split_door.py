"""把中心对称的门图从正中央切成左右两半，导出为 PNG（无损，适合导入 UE）。"""
from PIL import Image
import os

src = r"C:/Users/C1389/Desktop/Door.jpg"
out_dir = os.path.dirname(src)

img = Image.open(src).convert("RGBA")
w, h = img.size
mid = w // 2

left = img.crop((0, 0, mid, h))
right = img.crop((mid, 0, w, h))

left_path = os.path.join(out_dir, "Door_Left.png")
right_path = os.path.join(out_dir, "Door_Right.png")
left.save(left_path)
right.save(right_path)

print(f"原图: {w}x{h}, 从 x={mid} 处中分")
print(f"已保存: {left_path}  尺寸 {left.size}")
print(f"已保存: {right_path}  尺寸 {right.size}")
