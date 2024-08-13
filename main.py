from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
import uvicorn

client = OpenAI()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/completions", response_class=HTMLResponse)
async def completions(request: Request, prompt: str = Form(...)):
    """
    Chat Completions API を使用し、ユーザーの入力に対して回答を生成します。
    https://platform.openai.com/docs/guides/chat-completions
    """

    with open("context.txt", "r") as file:
        context = file.read()

    system_content = f"""
    あなたはエンジニアカフェのコミュニティーマネージャです。以下の情報を踏まえて回答してください：
    {context}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": prompt},
        ],
    )

    message = response.choices[0].message.content

    return templates.TemplateResponse(
        "index.html", {"request": request, "message": message}
    )


@app.post("/images", response_class=HTMLResponse)
async def dalle(request: Request, prompt: str = Form(...)):
    """
    Images API を使用し、ユーザーの入力に対して画像を生成します。
    https://platform.openai.com/docs/guides/images/usage?context=python
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""
                以下のユーザの入力に対して「細部まで緻密に描かれた高品質なアイコン画像」を
                出力するDALL-Eのプロンプトをできる限り詳細を含めて作成してください：
                {prompt}
                """,
            },
        ],
    )

    image_prompt = response.choices[0].message.content

    response = client.images.generate(
        model="dall-e-3", prompt=image_prompt, size="1024x1024", quality="standard", n=1
    )

    image_url = response.data[0].url

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "image_prompt": image_prompt, "image_url": image_url},
    )


if __name__ == "__main__":
    load_dotenv(override=True)
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
