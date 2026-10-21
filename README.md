# EmbeddingGemma 2 for Unity

Unityでテキスト・画像・音声の意味検索を行うためのライブラリです。Unity Sentisで各入力を共通の768次元ベクトルに変換し、日本語・英語で文書や画像、音声を検索できます。

**[Windows版 v0.1.0](https://github.com/ayutaz/unity-embeddinggemma-2/releases/tag/v0.1.0) 公開中。** モデルは別途取得が必要です。

## できること

- **テキスト検索** — 文書を登録し、意味の近い文章を検索。
- **画像検索** — Texture・RenderTextureからベクトルを生成し、文章で画像を検索。
- **音声検索** — mono PCM音声からベクトルを生成し、文章で音声を検索。

推論はSentisのGPUComputeを使用します。追加のネイティブ推論プラグインは不要です。

## 対応環境

- **Windows Editor / Player**、Unity **6000.3.21f1**、Sentis **2.6.1**。
- 実GPU検証環境：Direct3D 12 / NVIDIA RTX 4070 Ti SUPER。
- FP32 / Float16重みのモデルに対応。

macOS・iOS・Androidは未対応です。画像・音声には固定入力サイズ・長さの制約があり、GPU非対応時のCPUへの自動切り替えはありません。モデルは大きく、他のGPUでの最低要件は未確定です。[対応構成と制限](docs/windows-supported-configuration.md)を確認してください。

## 導入

1. Unityの **Window → Package Manager → Add package from git URL** を開き、次のURLを入力します。

   ```text
   https://github.com/ayutaz/unity-embeddinggemma-2.git?path=/Packages/com.ayutaz.embeddinggemma#v0.1.0
   ```

2. モデルを準備します。**パッケージにモデル重みは含まれません。**
   - テキストのみ：[テキストモデルの準備](docs/model-preparation.md)
   - 画像・音声も使う：[変換済みモデルbundleの取得](docs/media-model-preparation.md)
3. Package Managerの **Samples** からサンプルをImportし、各手順に従ってモデルを指定します。

## サンプル

まずは **Text Search** から試してください。

| サンプル | 内容 | 開くシーン |
| --- | --- | --- |
| [Text Search](Packages/com.ayutaz.embeddinggemma/Samples~/TextSearch/README.md) | 日本語・英語の文書検索 | `TextSearch.unity` |
| [Image and Audio Search](Packages/com.ayutaz.embeddinggemma/Samples~/MultimodalSearch/README.md) | 画像・音声の登録とテキストによる検索 | `MultimodalSearch.unity` |

## ドキュメント

- [C# API・使用例](Packages/com.ayutaz.embeddinggemma/Documentation~/index.md)
- [検証状況](docs/status.md)・[今後の計画](docs/roadmap.md)
- [ドキュメント一覧](docs/README.md)

## コントリビューション

不具合報告・改善提案は[Issues](https://github.com/ayutaz/unity-embeddinggemma-2/issues)、変更提案はPull Requestで受け付けます。不具合には環境と再現手順を添えてください。開発時は[開発規則](AGENTS.md)と[CIの手順](docs/ci.md)を参照してください。

## ライセンス

コードは[Apache License 2.0](LICENSE)です。モデルの利用条件は[モデルのライセンス](docs/model-license.md)を参照してください。
