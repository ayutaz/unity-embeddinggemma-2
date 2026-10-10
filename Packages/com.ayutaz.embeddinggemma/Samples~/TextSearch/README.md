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

準備メニューの再実行では、生成元commit・Unityバージョン・入力モデルとtokenizer・出力3ファイルのサイズとSHA-256がすべて一致すると、モデルの読み込み・再変換・保存を省きます。旧形式のreceiptや不一致は再生成します。大きいファイルの完全なhash確認は毎回必要です。
Windows Editorでは1MiB以上のファイルをWindows標準の`certutil.exe`でSHA-256照合し、利用できない場合は従来の計算へ戻ります。shellや追加のネイティブプラグインは不要です。他のOSと小さいファイルは従来の計算を使います。実モデルのキャッシュ再利用は変更後の単回測定でEditor内7.38秒でした。初回変換の時間、他環境、domain reloadの改善を示す測定ではありません。
既定の配置先は`Assets/StreamingAssets/EmbeddingGemmaTextSearch/`です。モデル・tokenizerと生成結果をGitへコミットしないでください。

取得・監査・新規consumer・更新・artifact失効時の手順は
[リポジトリのモデル準備手順](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/model-preparation.md)を参照してください。
現在の同期ファイル読み込みを、未検証のPlayer配置や端末で動作確認済みとは扱いません。
