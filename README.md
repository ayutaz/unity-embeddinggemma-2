# unity-embeddinggemma-2

Unity の推論ライブラリ [Sentis](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/index.html) だけを使って、Google の [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) を動かすプロジェクトです。テキスト・画像・音声を、同じ 768 次元の空間の埋め込みに変換できるようにします。

> **状態: 計画段階**
> 調査と方針の決定まで終わっています。実装はこれからです。

## ゴール

次の 3 つをすべて満たすことを目指します。詳しくは [docs/goal.md](docs/goal.md) を参照してください。

1. **Sentis だけで動かす**: llama.cpp や ONNX Runtime などのネイティブプラグインを使わない
2. **テキスト・画像・音声に対応する**: 3 つの入力を同じ空間の埋め込みにして、入力の種類をまたいだ検索ができる
3. **入力を GPU 上で処理する**: `RenderTexture` やマイクの音声を、CPU に読み戻さずに埋め込みにする

## マイルストーン

| # | 内容 | 状態 |
| --- | --- | --- |
| M1 | テキスト用モデルを Sentis で動かす | 未着手 |
| M2 | テキスト版を公開する | 未着手 |
| M3 | 画像用モデルに対応する | 未着手 |
| M4 | 音声用モデルに対応する | 未着手 |

## 動作環境

| 項目 | バージョン |
| --- | --- |
| Unity | 6000.3.19f1(URP 2D テンプレート) |
| Sentis(`com.unity.ai.inference`) | 2.6.1 を使う予定(まだプロジェクトに入っていない) |
| Python(モデルの変換用) | 3.13 以上、[uv](https://docs.astral.sh/uv/) で管理 |

## フォルダ構成

```
.
├── Assets/          Unity のアセット
├── Packages/        Unity のパッケージ設定
├── ProjectSettings/ Unity のプロジェクト設定
├── docs/            調査結果と設計のドキュメント
└── tools/           モデルを変換する Python プロジェクト(uv)
```

## Python 環境(tools/)

モデルの書き出しと、正しさを確かめるための参照データの作成に使います。Python は uv で管理します。

```sh
cd tools
uv sync                      # 依存をインストールする
uv run python <スクリプト>   # スクリプトを実行する
uv add <パッケージ>          # 依存を追加する(pip install は使わない)
```

主な依存は torch・transformers・sentence-transformers・onnx です。

## モデルファイルについて

モデルファイル(`.safetensors`、`.pt2`、`.onnx`、`.sentis` など)はリポジトリにコミットしません。Git LFS も使いません。モデルは Hugging Face から取得し、`tools/` のスクリプトで変換する予定です。

## ドキュメント

| ドキュメント | 内容 |
| --- | --- |
| [docs/goal.md](docs/goal.md) | ゴール、達成の基準、マイルストーン、対象外のこと |
| [docs/technical-approach.md](docs/technical-approach.md) | Sentis 2.6 の調査、EmbeddingGemma 2 の構造、モデルを持ち込む方法の比較、設計案、検証方法 |
| [docs/embeddinggemma-2-unity-novelty.md](docs/embeddinggemma-2-unity-novelty.md) | Unity 対応の新規性の調査(2026-10-07 時点) |

## ライセンス

このリポジトリは [Apache License 2.0](LICENSE) で公開します。

EmbeddingGemma 2 のモデルも Apache License 2.0 で配布されています。モデルファイルはこのリポジトリに含めません。
