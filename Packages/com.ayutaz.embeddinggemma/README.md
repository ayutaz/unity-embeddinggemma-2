# EmbeddingGemma Text

Unity SentisでEmbeddingGemma 2のテキスト埋め込みを作るUPMパッケージです。
開発版 `0.1.0-pre.1`。公開リリース・モバイル対応の受け入れはまだ完了していません。
2026-10-09確認: [PR #6](https://github.com/ayutaz/unity-embeddinggemma-2/pull/6)で実装・検証済み、mainへの反映待ちです。
別の空のUnityプロジェクトでローカルフォルダ依存の解決・compile・契約29件が成功しました。元の検証プロジェクトでは実モデルCPU / GPUComputeの回帰も成功しています。
新規consumerでの実モデルGPU、Git URL経由のEditor導入は未検証です。

- 検証対象: Unity 6000.3.16f1、Sentis 2.6.1、Windows Editor CPU / GPUCompute。
- API: `TextEmbedder`、`TextPrompts`、`TextTokenizer`、`TextModelFile`。
- batch 1 / length 128 / 768次元の固定モデルを使用。推論はメインスレッドの同期処理です。
- モデル重み・参照artifact・ネイティブプラグインは含みません。モデルのダウンロードも自動実行しません。
- 実行依存はSentisとUnity提供Newtonsoft JSON。検証用のuloop、URP、Pythonはパッケージの実行依存ではありません。

導入・使用例は [Documentation~/index.md](Documentation~/index.md)、実測・変換・検証手順は
[リポジトリ](https://github.com/ayutaz/unity-embeddinggemma-2)を参照してください。
検索サンプル、macOS / iOS / Androidの実モデル検証はM2の後続作業です。
