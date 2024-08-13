import base64

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, File, UploadFile
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


@app.post("/image-captioning", response_class=HTMLResponse)
async def image_captioning(request: Request, image: UploadFile = File(...)):
    """
    GPT4o-mini を使用し、画像の説明を生成します。
    https://cookbook.openai.com/examples/tag_caption_images_with_gpt4v
    """
    image_content = await image.read()
    image_base64 = base64.b64encode(image_content).decode("utf-8")
    image_data_url = (
        f"data:image/{image.content_type.split('/')[-1]};base64,{image_base64}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
                あなたは画像の中から標識やラベルなどのテキストを読み取り、正確に回答するAIです。
                """,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": "標識やラベルに書かれたテキストをJSONで出力してください。",
            },
        ],
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    image_caption = response.choices[0].message.content
    print(image_caption)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "image_caption": image_caption},
    )


if __name__ == "__main__":
    load_dotenv(override=True)
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
