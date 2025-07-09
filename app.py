from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from classify import classify_code, _base_window_sizes

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    inputs = ""
    for t, v in _base_window_sizes.items():
        key = t.replace(" ", "_")               # HTML name 无空格
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
    request: Request,                          # ← 用 Request 拿表单
    file: UploadFile = File(...),
    size_ratio: float = Form(1.2),
):
    # ------------- 收集 per-topic window -------------
    form = await request.form()
    custom = {}
    for k, v in form.items():
        if k in {"file", "size_ratio"}:
            continue
        topic = k.replace("_", " ")
        try:
            custom[topic] = int(v)
        except ValueError:
            pass                                    # 忽略空/非法输入

    # ------------- 推理并返回 -------------------------
    code = (await file.read()).decode("utf-8", errors="ignore")
    html = classify_code(code, size_ratio=size_ratio, custom_windows=custom)
    return HTMLResponse(html)

