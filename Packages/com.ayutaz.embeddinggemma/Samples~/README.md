# サンプル

Package ManagerでEmbeddingGemmaを選び、Samplesから目的のサンプルをImportします。モデルは同梱・自動取得しません。

| サンプル | 内容 | 必要な準備 |
| --- | --- | --- |
| [Text Search](TextSearch/README.md) | 日本語・英語の固定文書をテキストで検索 | `.sentis`と`tokenizer.json` |
| [Image and Audio Search](MultimodalSearch/README.md) | 実画像・mono PCMを登録し、テキストで横断検索 | 監査済みbundle、対応する固定入力profile |

まず[導入・モデル準備](../Documentation~/index.md)を確認してください。画像・音声には[変換済みモデルbundleの取得手順](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/media-model-preparation.md)があります。両sampleの主経路はGPUComputeで、GPU非対応時にCPUへ自動切り替えしません。

Windowsの固定source検証と他環境の未達条件は[検証状況](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/status.md)を参照してください。
