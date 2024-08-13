# OpenAI FastAPI example

生成AIワークショップ用の実装です。

## 環境構築

1. [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) をインストール
1. 仮想環境を構築
    ```sh
    conda env create -f environment.yml
    conda activate fastapi
    ```
1. `.env.example`をコピーして`.env`を作成し、環境変数を設定

## 使い方

1. サーバを起動
    ```sh
    python main.py
    ```
1. [http://localhost:8000](http://localhost:8000) にアクセス
