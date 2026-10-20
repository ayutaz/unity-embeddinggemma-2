# EmbeddingGemma for Unity

Unity Sentisでテキスト・画像・音声を共通の正規化768次元ベクトルへ変換するUPMパッケージです。テキストによる文書・画像・音声の意味検索に使えます。

**`0.1.0` の公開候補です。正式tag・Releaseは未公開です。** モデルは同梱せず、パッケージ導入だけでは取得しません。画像・音声・テキストを含む全6 moduleは[実験用モデルpreviewの取得手順](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/media-model-preparation.md)からFP32 / Float16重みの変換済みbundleを明示取得できます。descriptor・全objectのhashを照合し、LICENSE・NOTICE・MODIFICATIONSを保持してください。previewの公開・取得成功は、全環境・性能・正式SDK公開の合格とは別です。

## 必要な環境

最初の正式リリースはWindows Editor / Playerを対象に準備しています。Windows候補の機能検証は、Editorの実モデルと同一SDK実装の既存Player結果で完了しました。性能の条件別記録と初回対応構成は整理済みで、新CI・新tagからの新規導入・正式公開は残っています。macOS / iOS / Androidは今後の対応で、現時点の正式対応環境には含めません。

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
