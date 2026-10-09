# EmbeddingGemma Text

Unity SentisでEmbeddingGemma 2のテキスト埋め込みを作るUPMパッケージです。
開発版 `0.1.0-pre.1`。公開リリース・モバイル対応の受け入れはまだ完了していません。
2026-10-10確認: UPM構成は[PR #6](https://github.com/ayutaz/unity-embeddinggemma-2/pull/6)、検索サンプル・モデル準備はPR #11でmainへ統合済みです。PR #12統合後の確認基準main `4ee0cbb`のCI全8 job成功を確認しています。
別の空のUnityプロジェクトでローカルフォルダ依存の解決・compile・契約29件が成功しました。元の検証プロジェクトでは実モデルCPU / GPUComputeの回帰も成功しています。
新規consumerでも実モデル4条件の全順位とベクトル、CLI 4 passed / skip 0、CPUの日英検索・空入力・解放を確認しました。Git URL経由のEditor導入、他環境は未検証です。

- 検証対象: Unity 6000.3.16f1、Sentis 2.6.1、Windows Editor CPU / GPUCompute。
- API: `TextEmbedder` / `ITextEmbedder`、`TextPrompts`、`TextTokenizer`、`TextModelFile`、`TextSearchIndex`、`TextSearchSession`、`SearchDocument`、`SearchHit`。
- batch 1 / length 128 / 768次元の固定モデルを使用。推論はメインスレッドの同期処理です。
- モデル重み・参照artifact・ネイティブプラグインは含みません。モデルのダウンロードも自動実行しません。
- 実行依存はSentisとUnity提供Newtonsoft JSON。検証用のuloop、URP、Pythonはパッケージの実行依存ではありません。

導入・使用例は [Documentation~/index.md](Documentation~/index.md)、実測・変換・検証手順は
[リポジトリ](https://github.com/ayutaz/unity-embeddinggemma-2)を参照してください。
Text Searchサンプルとモデル準備メニューを[PR #11](https://github.com/ayutaz/unity-embeddinggemma-2/pull/11)で追加しました。Package Managerからインポートし、[サンプル手順](Samples~/TextSearch/README.md)に沿ってモデルを配置します。Windowsの元プロジェクトと新規consumerでfp32 / Float16重み × CPU / GPUComputeの全4条件・全順位一致と画面操作を確認しました。準備キャッシュのTDD18件も合格。macOS / iOS / Android・Git URL導入・公開リリースは残っています。
