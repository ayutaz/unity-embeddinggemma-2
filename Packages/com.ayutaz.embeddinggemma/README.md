# EmbeddingGemma（開発版）

> 2026-10-16: Draft PR #81の実GPU検証で、画像・音声の各18条件が実入力から最終768次元まで合格しています。音声は分割encoderと`AudioEmbedder.EmbedAsync`を使用します。必須CIの音声ONNX比較は失敗中で、main未統合・未リリースです。source別証拠と後続工程は[現在の状態](../../docs/status.md)を参照してください。

Unity SentisでEmbeddingGemma 2を動かす開発版UPMです。テキストの既存受け入れと、画像・音声の開発中APIを区別します。
開発版 `0.1.0-pre.1`。公開リリース・モバイル対応の受け入れはまだ完了していません。
2026-10-10確認: Windowsの新しい空のUnityプロジェクトへ、commit `01f3d8682a97053bf4583e14fd8fbed14e2c8cae`のGit URLで導入し、requested / resolved SHA、配布source、compile、UPM 38 / sample EditMode 64 / PlayMode 2件を確認しました。
実モデルFP32 / Float16重み × CPU / GPUComputeの全4条件で、固定6文書 / 4queryの全順位・scoreがPython参照と一致しました。Game Viewで日英検索、空入力、解放、warm cache、Play停止も確認しています。
条件・数値・CLI接続失敗から同じ実行のXMLを回収した経緯は[固定Git consumerの検証記録](https://github.com/ayutaz/unity-embeddinggemma-2/blob/b1a8c67c62066bfa5e1830a947563ace3dff9fa5/docs/m2-latest-git-consumer-validation.md)を参照してください。Windows以外、正式候補 / tag固定導入、正式リリースは未完了です。allocation Logとfont警告も追跡中です。

- 検証対象: Unity 6000.3.16f1、Sentis 2.6.1、Windows Editor CPU / GPUCompute。
- 既存テキストAPI: `TextEmbedder` / `ITextEmbedder`、`TextPrompts`、`TextTokenizer`、`TextModelFile`、`TextSearchIndex`、`TextSearchSession`、`SearchDocument`、`SearchHit`。
- DraftマルチモーダルAPI: `ImageEmbedder.EmbedAsync` / `AudioEmbedder.EmbedAsync`と分割GPU encoder。音声の[使用例・所有権](Documentation~/audio-embedder.md)を参照してください。
- テキストはbatch 1 / length 128 / 768次元の固定モデルを使用。テキスト推論はメインスレッドの同期処理です。
- モデル重み・参照artifact・ネイティブプラグインは含みません。モデルのダウンロードも自動実行しません。
- 実行依存はSentisとUnity提供Newtonsoft JSON。検証用のuloop、URP、Pythonはパッケージの実行依存ではありません。

導入・使用例は [Documentation~/index.md](Documentation~/index.md)、実測・変換・検証手順は
[リポジトリ](https://github.com/ayutaz/unity-embeddinggemma-2)を参照してください。
Package ManagerからText Searchサンプルをインポートし、[サンプル手順](Samples~/TextSearch/README.md)に沿ってモデルを配置します。[PR #24](https://github.com/ayutaz/unity-embeddinggemma-2/pull/24)で選択した重みだけを保持するcacheを追加し、Windowsではfile URLの初回展開・warm完全hashと実GPU検索を確認しました。Androidのjar経路は実装済みで、旧sourceのAPK build / payload監査は成功しています。Android実機と最新candidateのsample Playerは未検証です。model deserializeと推論はメインスレッドの同期処理です。
