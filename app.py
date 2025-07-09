from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from classify import classify_code, _base_window_sizes  # 导入默认窗口

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    inputs = ""
    for t, v in _base_window_sizes.items():
        key = t.replace(" ", "_")          # name 里不能有空格
        inputs += f'{t}: <input name="{key}" type="number" value="{v}" min="1"><br>'
    return f"""
    <h3>Upload C++ File</h3>
    <form action="/classify" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".cpp,.h,.hpp" required><br>
      size_ratio: <input name="size_ratio" type="number" step="0.1" value="1.2"><br>
      {inputs}
      <button>Upload</button>
    </form>
    """

@app.post("/classify", response_class=HTMLResponse)
async def classify(
    file: UploadFile = File(...),
    size_ratio: float = Form(1.2),
    # 动态收集所有以主题名替换空格后的字段
    **windows
):
    # 还原 key → topic 名
    custom = {
        k.replace("_", " "): int(v)
        for k, v in windows.items()
        if k not in {"file", "size_ratio"} and v  # 排除其他字段
    }
    code = (await file.read()).decode("utf-8", errors="ignore")
    html = classify_code(code, size_ratio=float(size_ratio), custom_windows=custom)
    return HTMLResponse(html)
