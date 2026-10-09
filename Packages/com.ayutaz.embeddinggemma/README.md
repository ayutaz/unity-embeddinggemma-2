# EmbeddingGemma Text

Unity SentisでEmbeddingGemma 2のテキスト埋め込みを作るUPMパッケージです。
開発版 `0.1.0-pre.1`。公開リリース・モバイル対応の受け入れはまだ完了していません。
2026-10-10確認: [PR #6](https://github.com/ayutaz/unity-embeddinggemma-2/pull/6)でmainへ統合済みです。PR #8統合後のmain CIも全8 job成功しています。
別の空のUnityプロジェクトでローカルフォルダ依存の解決・compile・契約29件が成功しました。元の検証プロジェクトでは実モデルCPU / GPUComputeの回帰も成功しています。
新規consumerでの実モデルGPU、Git URL経由のEditor導入は未検証です。

- 検証対象: Unity 6000.3.16f1、Sentis 2.6.1、Windows Editor CPU / GPUCompute。
- API: `TextEmbedder` / `ITextEmbedder`、`TextPrompts`、`TextTokenizer`、`TextModelFile`、`TextSearchIndex`、`TextSearchSession`、`SearchDocument`、`SearchHit`。
- batch 1 / length 128 / 768次元の固定モデルを使用。推論はメインスレッドの同期処理です。
- モデル重み・参照artifact・ネイティブプラグインは含みません。モデルのダウンロードも自動実行しません。
- 実行依存はSentisとUnity提供Newtonsoft JSON。検証用のuloop、URP、Pythonはパッケージの実行依存ではありません。

導入・使用例は [Documentation~/index.md](Documentation~/index.md)、実測・変換・検証手順は
[リポジトリ](https://github.com/ayutaz/unity-embeddinggemma-2)を参照してください。
このブランチではText Searchサンプルとモデル準備メニューを[PR #11](https://github.com/ayutaz/unity-embeddinggemma-2/pull/11)で追加しました。Package Managerからインポートし、[サンプル手順](Samples~/TextSearch/README.md)に沿って明示的にモデルを配置します。Windowsでは保存済みfp32 / Float16重み × CPU / GPUComputeの全4条件で固定queryの全順位が一致しました。mainへの統合、画面操作・新規consumer実モデル再現、macOS / iOS / Androidの受け入れと公開リリースは未完了です。
