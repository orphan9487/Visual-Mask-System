import requests

# 修正後的路徑：位於 text_to_image 資料夾內
url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py"

print("🚀 正在從最新路徑下載訓練腳本...")
try:
    r = requests.get(url)
    r.raise_for_status() # 如果還是 404 會直接報錯，不會存成 HTML
    with open("train_text_to_image_lora.py", "wb") as f:
        f.write(r.content)
    print("✅ 下載成功！請打開檔案確認第一行是 'import argparse'。")
except Exception as e:
    print(f"❌ 下載失敗，錯誤代碼：{e}")