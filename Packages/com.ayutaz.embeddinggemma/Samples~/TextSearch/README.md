# Text Search

別途用意したEmbeddingGemma 2のSentisモデルで、日本語 / 英語の固定6文書を検索するサンプルです。
モデル・tokenizerは含まれず、自動ダウンロードしません。Unity 6000.3.16f1 / Sentis 2.6.1を対象にしています。

1. 成功したActionsの固定参照を`stage --search`で監査・配置する。
2. Editorの**Tools → EmbeddingGemma → Prepare Text Search Models**でfp32 / Float16重みの`.sentis`とtokenizerを用意する。
3. `TextSearch.unity`を開いてPlayし、モデルとtokenizerのパス・backendを指定する。
4. **モデルと文書を準備**、検索文の入力、**検索**で順位とcosineを表示する。
5. **解放 / モデルを変更**で再準備できる。Play停止・無効化時にも推論器を解放する。

モデル未準備・空の検索文・GPU非対応・推論失敗は画面で確認できます。CPUへの自動切り替えはありません。
文書は準備時に一度だけ埋め込み、検索時はqueryだけを推論します。同点順位は文書IDのordinal順です。
既定の配置先は`Assets/StreamingAssets/EmbeddingGemmaTextSearch/`です。モデル・tokenizerと生成結果をGitへコミットしないでください。

取得・監査・新規consumer・更新・artifact失効時の手順は
[リポジトリのモデル準備手順](https://github.com/ayutaz/unity-embeddinggemma-2/blob/feat/m2-text-search-sample/docs/model-preparation.md)を参照してください。
現在の同期ファイル読み込みを、未検証のPlayer配置や端末で動作確認済みとは扱いません。
