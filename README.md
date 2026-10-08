# unity-embeddinggemma-2

Unity の推論ライブラリ [Sentis](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/index.html) だけを使って、Google の [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) を動かすプロジェクトです。テキスト・画像・音声を、同じ 768 次元の空間の埋め込みに変換できるようにします。

> **状態: M1 実装中**
> 2026-10-08: 参照生成・Core ATen export・uloop ハーネスを [PR #1](https://github.com/ayutaz/unity-embeddinggemma-2/pull/1) で main（`60f906d`）に統合しました。
> 統合後の main CI ではPython 4環境の各47テストとworkflow lintが成功。最終PR CIでは実モデルの保存済み `.pt2` の全15ケース照合も成功しました。
> ローカル Unity 6000.3.16f1 に Unity CLI Loop を導入し、依存解決・テストのコンパイルに成功しました。自動検証ハーネスを追加しています。
> 未マージの [PR #3](https://github.com/ayutaz/unity-embeddinggemma-2/pull/3) では実モデルの配置と空文字互換修正後、Windowsのtokenizer・fp32 CPU / GPUCompute・C# APIが全15ケース合格しました。mainへの統合、`.sentis`保存・fp16量子化・時間 / メモリ測定が残ります。
> クラウドUnity CIはSecretsが必要なLinux CPUの手動補助検証です（未実行）。
> リポジトリは public ですが、UPM パッケージのリリースはまだ行っていません。

## ゴール

次の 3 つをすべて満たすことを目指します。詳しくは [docs/goal.md](docs/goal.md) を参照してください。

1. **Sentis だけで動かす**: llama.cpp や ONNX Runtime などのネイティブプラグインを使わない
2. **テキスト・画像・音声に対応する**: 3 つの入力を同じ空間の埋め込みにして、入力の種類をまたいだ検索ができる
3. **入力を GPU 上で処理する**: `RenderTexture` やマイクの音声を、CPU に読み戻さずに埋め込みにする

## マイルストーン

| # | 内容 | 状態 |
| --- | --- | --- |
| M1 | テキスト用モデルを Sentis で動かす | PR #3でfp32・API合格、未統合。保存・量子化・測定が残る |
| M2 | テキスト版の UPM パッケージとサンプルをリリースする | 未着手 |
| M3 | 画像用モデルに対応する | 未着手 |
| M4 | 音声用モデルに対応する | 未着手 |

## 動作環境

| 項目 | バージョン |
| --- | --- |
| Unity | 6000.3.16f1(URP 2D テンプレート) |
| Sentis(`com.unity.ai.inference`) | 2.6.1（ローカル Editor で依存解決・コンパイル済み） |
| Unity CLI Loop | パッケージ 3.14.0 / dispatcher 3.8.1 / project runner 3.8.0 |
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
uv sync --locked             # lockfile と同じ依存をインストールする
uv run --locked python <スクリプト> # スクリプトを実行する
uv add <パッケージ>          # 依存を追加する(pip install は使わない)
```

主な依存は torch・transformers・sentence-transformers・onnx です。

開発は TDD で行い、重い実行は GitHub Actions を利用します。`main` へ直接 push せず、作業ブランチから PR を作成します。
CI の構成と Unity ライセンスの準備は [docs/ci.md](docs/ci.md)、M1 の詳細手順と進捗は [docs/m1-plan.md](docs/m1-plan.md) を参照してください。

Unity のローカル自動操作とハーネスの導入・実行方法は [docs/automation.md](docs/automation.md) を参照してください。

小さいモデルを使うオフライン単体テスト:

```sh
cd tools
uv run --locked pytest -q
```

GitHub Actions が実行する参照生成・変換コマンド:

```sh
uv run --locked python -m embeddinggemma_tools prepare --output ../artifacts/m1
```

モデルの revision を固定して参照データ・設定済み tokenizer・`.pt2` を生成し、保存後のモデルを参照実装と比較します。
この Python 側の一致と、Sentis 側の M1 合格は別々に検証します。

基盤の [PR #1](https://github.com/ayutaz/unity-embeddinggemma-2/pull/1) はマージ済みです。
次は PR #2 / #3の最新CI・依存関係を確認し、`.sentis`保存・再読み込み → fp16量子化 → 時間・メモリ測定を進めます。詳細な開始条件と合格基準は [M1計画](docs/m1-plan.md) を参照してください。
PR #3で追加した [C# API手順](docs/runtime-api.md)、[実モデル実行記録](docs/m1-runtime-validation.md)、[詳細作業計画](docs/m1-runtime-plan.md) も参照してください。
互換性修正と C# API を TDD で進め、その後 `.sentis` 保存・fp16 量子化・時間 / メモリ測定を行います。
新しい実装も作業ブランチ / PR を使い、マージは依頼があるまで行いません。
詳細な順序は [計画](docs/m1-plan.md)、CI の確認記録は [検証記録](docs/m1-validation.md)を参照してください。
ローカル検証に GitHub Secrets は不要です。任意の Linux CPU 手動 CI を使う場合だけ準備します。

## モデルファイルについて

モデルファイル(`.safetensors`、`.pt2`、`.onnx`、`.sentis` など)はリポジトリにコミットしません。Git LFS も使いません。`tools/` の CLI が Hugging Face の固定 revision から取得して変換します。実モデルのPython export照合はCIで成功し、Sentisでの実行確認はこれからです。

## ドキュメント

| ドキュメント | 内容 |
| --- | --- |
| [docs/goal.md](docs/goal.md) | ゴール、達成の基準、マイルストーン、対象外のこと |
| [docs/m1-plan.md](docs/m1-plan.md) | M1 の詳細計画、TDD の進め方、検証状況 |
| [docs/m1-validation.md](docs/m1-validation.md) | TDD の red / green、最新テスト結果、未検証項目、外部状態の確認記録 |
| [docs/ci.md](docs/ci.md) | GitHub Actions、PR 運用、Unity CI の準備 |
| [docs/technical-approach.md](docs/technical-approach.md) | Sentis 2.6 の調査、EmbeddingGemma 2 の構造、モデルを持ち込む方法の比較、設計案、検証方法 |
| [docs/embeddinggemma-2-unity-novelty.md](docs/embeddinggemma-2-unity-novelty.md) | Unity 対応の新規性の調査(2026-10-07 時点) |

## ライセンス

このリポジトリは [Apache License 2.0](LICENSE) で公開します。

EmbeddingGemma 2 のモデルも Apache License 2.0 で配布されています。モデルファイルはこのリポジトリに含めません。
