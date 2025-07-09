from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from classify import classify_code

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home():
    return """
      <h3>Upload C++ File</h3>
      <form action="/classify" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".cpp,.h,.hpp" required>
        <button>Upload</button>
      </form>
    """

# ⬇ 只保留 File(...)，不要写 max_length
@app.post("/classify", response_class=HTMLResponse)
async def classify(file: UploadFile = File(...)):
    code = (await file.read()).decode("utf-8", errors="ignore")
    html = classify_code(code)
    return HTMLResponse(html)
