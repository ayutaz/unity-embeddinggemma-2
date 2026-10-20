# EmbeddingGemma for Unity

Unity Sentisでテキスト・画像・音声を共通の正規化768次元ベクトルへ変換するUPMパッケージです。テキストによる文書・画像・音声の意味検索に使えます。

**[Windows版 v0.1.0](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0) 公開中。** モデル重みは同梱しません。[モデルの準備手順](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/media-model-preparation.md)からFP32 / Float16重みの変換済みbundleを取得してください。

## 必要な環境

対応環境はWindows Editor / Playerです。macOS / iOS / Androidは未対応です。

- 基準Unity **6000.3.21f1**、Sentis **2.6.1**。UPMの最小宣言は6000.3.16f1。
- Runtime依存：`com.unity.ai.inference`、`com.unity.nuget.newtonsoft-json` **3.2.2**（自動解決）。
- Windows / D3D12 / RTX4070 Ti SUPERで固定sourceの実GPU検証済み。他OS・モバイルの受け入れは未完了。
- Python・uloop・URPはRuntime依存ではありません。

## 利用する

1. [導入・API](Documentation~/index.md)を読み、同じrevisionのモデル・tokenizerを準備します。
2. Package Managerで **Text Search** または **Image and Audio Search** をImportします。
3. サンプルの手順に従ってモデルを指定し、sceneを開いてPlayします。

| API・サンプル | 内容 |
| --- | --- |
| `TextEmbedder` / [Text Search](Samples~/TextSearch/README.md) | query / document / raw text、保存、テキスト検索 |
| `ImageEmbedder.EmbedAsync` | GPU Texture / RenderTexture入力 |
| `AudioEmbedder.EmbedAsync` | 固定16kHz mono PCM入力。非同期前処理・分割encoder |
| [Image and Audio Search](Samples~/MultimodalSearch/README.md) | 実画像・音声の登録、テキスト検索、cache・キャンセル・切替 |

メインスレッドから使用し、借用stageは非同期処理完了まで保持してください。GPU非対応時のCPUへの自動切り替えはありません。画像・音声には固定入力profileの制約があります。

詳しい対応範囲・Windows Playerの条件・検証sourceは[検証状況](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/status.md)、残作業は[ロードマップ](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/roadmap.md)を参照してください。コードは[Apache-2.0](LICENSE.md)です。
