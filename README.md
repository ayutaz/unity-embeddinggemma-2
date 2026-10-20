# EmbeddingGemma 2 for Unity

Unity Sentisで、テキスト・画像・音声を共通の768次元ベクトルへ変換するライブラリです。日本語・英語のテキスト検索や、文章で画像・音声を探す検索機能をUnityに組み込めます。

モデル推論にはSentisを使い、追加のネイティブ推論プラグインは不要です。画像・音声はGPUで前処理から推論まで実行します。トークナイズ・ファイルの読み込みとデコード・PCMの取得はCPU側で行い、最終ベクトルはCPU配列として返します。

**開発版 `0.1.0-pre.1`。最初の正式リリースはWindows Editor / Playerを対象に準備しています。正式リリースはまだありません。** モデルは別途準備が必要です。macOS・iOS・Androidは今後の対応で、未検証です。[残る検証と公開条件](docs/next-release-plan.md)を参照してください。

[導入・API](Packages/com.ayutaz.embeddinggemma/Documentation~/index.md) · [モデル準備](docs/model-preparation.md) · [現在の対応範囲](docs/status.md) · [ロードマップ](docs/roadmap.md) · [ドキュメント一覧](docs/README.md)

## できること

| 入力 | API・サンプル | 主な用途 |
| --- | --- | --- |
| テキスト | `TextEmbedder` / **Text Search** | 文書の意味検索、日本語・英語の検索 |
| 画像 | `ImageEmbedder.EmbedAsync` / **Image and Audio Search** | Texture・RenderTextureの埋め込み、文章で画像を検索 |
| 音声 | `AudioEmbedder.EmbedAsync` / **Image and Audio Search** | mono PCMの埋め込み、文章で音声を検索 |

画像・音声のベクトルは同じ索引へ登録できます。サンプルはcosine類似度による全件比較を使います。



## 動作環境

| 項目 | 現在の基準 |
| --- | --- |
| Unity | **6000.3.21f1**（リポジトリの検証プロジェクト） |
| 推論 | Sentis / `com.unity.ai.inference` **2.6.1** |
| 実行依存 | Unity提供Newtonsoft JSON **3.2.2** |
| 実GPU検証 | Windows / Direct3D 12 / RTX 4070 Ti SUPER |
| モデル準備・開発 | Python **3.13以上**、[uv](https://docs.astral.sh/uv/)、GitHub CLI |

UPMの宣言上の最小版は6000.3.16f1です。これは対応する全Unity版・GPUの検証を意味しません。Python・uloop・URPは、利用側のRuntime依存ではありません。

## 導入

1. Unityの **Window → Package Manager** を開き、**Add package from git URL** を選びます。
2. 次のURLを入力します。PR #94統合後、Windowsの実行検証を確認した開発版へ固定しています。

   ```text
   https://github.com/ayutaz/unity-embeddinggemma-2.git?path=/Packages/com.ayutaz.embeddinggemma#186ffc1b330c0b609fef66cf4abfaeed5011fcbc
   ```

3. モデルを準備します。**パッケージを追加しただけでは推論できません。** Text Searchは[テキストモデルの取得・変換手順](docs/model-preparation.md)を、Image and Audio Searchは[全6 moduleのモデルpreview取得手順](docs/media-model-preparation.md)を使います。
4. Package ManagerのSamplesから目的のサンプルをImportします。

Package Managerでは現在 **EmbeddingGemma** と表示されます。このパッケージにテキスト・画像・音声のAPIと両サンプルを含みます。

| サンプル | 始め方 |
| --- | --- |
| **Text Search** | [手順](Packages/com.ayutaz.embeddinggemma/Samples~/TextSearch/README.md)に従い、モデルとtokenizerを指定して`TextSearch.unity`を再生します。まずこちらから試してください。 |
| **Image and Audio Search** | [手順](Packages/com.ayutaz.embeddinggemma/Samples~/MultimodalSearch/README.md)に従い、監査済みのモデルbundleを指定して`MultimodalSearch.unity`を再生します。 |

画像・音声・テキストを含む全6 moduleの変換済みbundleは、FP32 / Float16重みの[実験用モデルpreview](docs/media-model-preparation.md)から取得できます。固定descriptorのSHA256と全objectのhashを照合し、LICENSE・NOTICE・MODIFICATIONSも保持してください。空cacheからの匿名取得・中断復帰・破損修復とWindows GPU実行は[固定sourceで確認済み](docs/results/public-model-consumer-8649c94-20261018.json)です。元checkpointからの再変換、全環境の受け入れ、正式SDK Releaseとは別の経路・条件です。

既存プロジェクトにAPIを組み込む場合は、[C#の使用例とリソース管理](Packages/com.ayutaz.embeddinggemma/Documentation~/index.md)を参照してください。

## 現在の制約

- 画像・音声は固定profileを使用します。サンプルの対応寸法・音声長・形式は[入力仕様](Packages/com.ayutaz.embeddinggemma/Samples~/MultimodalSearch/README.md)を参照してください。任意の画像・音声長への対応は保証しません。
- GPU非対応時にCPUへ自動切り替えはしません。テキストAPIには明示的なCPU選択肢もありますが、画像・音声サンプルはGPU専用です。
- Unityのメインスレッドから使用します。非同期APIでもすべてのロード・処理が別スレッドになるわけではありません。
- モデルは大きく、メモリと初回ロードの負荷があります。固定検証のホストメモリ測定は[検証状況](docs/status.md)に記載しています。VRAMの必要量や他端末の最低要件は未確定です。
- Windowsでは固定sourceで実マイク取得・GPU検索、音声再生 / 停止、画像・音声の混在Playerを検証しています。手動操作全体・権限拒否・他OS・モバイル実機は未完了です。実行sourceと確認範囲は[検証状況](docs/status.md)を参照してください。

確認済みの結果と未検証の範囲は[検証状況](docs/status.md)、残作業の順序と完了条件は[ロードマップ](docs/roadmap.md)へまとめています。

## 開発・コントリビューション

不具合報告・改善提案は[Issues](https://github.com/ayutaz/unity-embeddinggemma-2/issues)、変更提案はPull Requestで受け付けます。不具合にはUnity / Sentis版、OS・GPU・graphics API、再現手順とログを添えてください。

Pythonの開発環境とテストは次の手順です。これは実モデル・Unity GPU検証とは別の単体テストです。

```sh
cd tools
uv sync --locked
uv run --locked pytest -q
```

振る舞いの変更はTDDで進め、重い変換・回帰はGitHub Actionsを優先します。作業ブランチからPRを作成してください。モデルや大きい生成物はコミットしません。詳細は[開発規則](AGENTS.md)・[CI](docs/ci.md)・[自動操作](docs/automation.md)を参照してください。

## ライセンス

コードは[Apache License 2.0](LICENSE)です。モデル重みは同梱しません。モデルの取得元・固定revision・利用条件は[モデルのライセンス確認](docs/model-license.md)を参照してください。
