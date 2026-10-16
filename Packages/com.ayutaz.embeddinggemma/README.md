# EmbeddingGemma（開発版）

> 2026-10-16: Draft PR #81の実GPU検証で、画像・音声の各18条件が実入力から最終768次元まで合格しています。音声は分割encoderと`AudioEmbedder.EmbedAsync`を使用します。必須CIの音声ONNX比較は失敗中で、main未統合・未リリースです。source別証拠と後続工程は[現在の状態](../../docs/status.md)を参照してください。

Unity SentisでEmbeddingGemma 2を動かす開発版UPMです。テキストの既存受け入れと、画像・音声の開発中APIを区別します。
開発版 `0.1.0-pre.1`。公開リリース・モバイル対応の受け入れはまだ完了していません。
2026-10-10確認: Windowsの新しい空のUnityプロジェクトへ、font寿命修正を含むcommit `fc66af7bc9150ddcf759917a5e9c92b2c5afa991`のGit URLで導入し、requested / resolved SHA、配布source、compile、UPM 38 / sample EditMode 66 / PlayMode 2件を確認しました。
実モデルFP32 / Float16重み × CPU / GPUComputeの全4条件で、固定6文書 / 4queryの全順位・scoreがPython参照と一致しました。Game Viewで日英検索、空入力、解放、warm cache、Play停止も確認しています。
後続のEditor測定fixture 2ファイルの修正ではsample EditMode 67件が合格し、PR #30で統合しました。固定Git導入全体の再実行とは分けています。[導入・実測と制約](https://github.com/ayutaz/unity-embeddinggemma-2/blob/6f55a8574df1bc75d51fb5992fd70dc7d0512fb9/docs/m2-font-git-consumer-validation.md)。Windows以外、正式候補 / tag導入、正式リリース、Editor検索DB / allocation / domain reloadは未完了です。

- 検証対象: Unity 6000.3.16f1、Sentis 2.6.1、Windows Editor CPU / GPUCompute。
- 既存テキストAPI: `TextEmbedder` / `ITextEmbedder`、`TextPrompts`、`TextTokenizer`、`TextModelFile`、`TextSearchIndex`、`TextSearchSession`、`SearchDocument`、`SearchHit`。
- DraftマルチモーダルAPI: `ImageEmbedder.EmbedAsync` / `AudioEmbedder.EmbedAsync`と分割GPU encoder。音声の[使用例・所有権](Documentation~/audio-embedder.md)を参照してください。
- テキストはbatch 1 / length 128 / 768次元の固定モデルを使用。テキスト推論はメインスレッドの同期処理です。
- モデル重み・参照artifact・ネイティブプラグインは含みません。モデルのダウンロードも自動実行しません。
- 実行依存はSentisとUnity提供Newtonsoft JSON。検証用のuloop、URP、Pythonはパッケージの実行依存ではありません。

導入・使用例は [Documentation~/index.md](Documentation~/index.md)、実測・変換・検証手順は
[リポジトリ](https://github.com/ayutaz/unity-embeddinggemma-2)を参照してください。
Package ManagerからText Searchサンプルをインポートし、[サンプル手順](Samples~/TextSearch/README.md)に沿ってモデルを配置します。Windowsではfile URLの初回展開・warm完全hashと実GPU検索を確認済み。Androidのjar経路は実装済みで実APK build・監査も成功しましたが、実機は未実行です。21f1の通常表示sample Playerでは日英GUI / GPU検索・解放・独立2起動の正常終了を確認しました。検証consumerの環境補修と新candidate導入・Float16/GPU性能・他OS実機のgateが残り、[PR #31](https://github.com/ayutaz/unity-embeddinggemma-2/pull/31)はdraft / 未統合です。model deserializeと推論はメインスレッドの同期処理です。
