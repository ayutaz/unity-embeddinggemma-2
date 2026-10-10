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
## Playerのローカルモデル配置

通常のfilesystemパスは従来の同期読み込みを使います。AndroidのStreamingAssets等の`jar:file://...!/assets/...`と`file://`は、**モデルと文書を準備**からcoroutineで所有cacheへ展開します。インターネット上のURLや、モデルと異なるdirectoryのtokenizerは受け付けません。

準備済みの`preparation.json`、選択した`model-fp32.sentis`または`model-float16.sentis`、`tokenizer.json`が必要です。receiptは成功、Unity version、固定形式の生成元commit / 元モデルSHA-256、固定3ファイルの長さとhashを要求します。選択した重みとtokenizerだけをコピーし、完全SHA-256を照合してからロードします。再準備でも毎回完全hashを確認し、同じreceiptと選択ならmanifestだけを転送します。

cacheは`Application.persistentDataPath/EmbeddingGemmaTextSearch/ModelCache`に置きます。成功後は選択した重み1つを含む`active`だけ、更新中は旧`active`と候補を保持します。失敗時は旧cacheを残し、中断した候補は次の試行で破棄します。未知のファイル・directoryは自動削除しません。worker使用中はlockを保持し、**解放 / モデルを変更**・無効化・Play停止でworkerとlockを解放します。展開中の解放は通信を中断し、後からworkerを作りません。

sampleを最小consumerへ導入する場合は、manifestのdependenciesに`"com.unity.modules.unitywebrequest": "1.0.0"`を有効にしてください。リポジトリのconsumer作成CLIの`--sample`はこれを追加します。配布Core Runtimeの直接依存は変更していません。

WindowsのRuntime hashはOSのCNGを使い、利用できない場合・他OSは完全.NET SHA-256へ戻ります。追加のネイティブDLLを配布しません。通信はcoroutineですが、hash・モデルdeserialize・文書埋め込みは同期処理です。全準備をフレーム予算内で処理する実装やbyte-range転送再開の保証ではありません。

Windows Editorのfile URL転送・実モデルGPU準備・検索・cache再利用を確認しました。Android APK / 実機、macOS / iOS、Playerのsample画面は未検証であり、この実装を全端末対応済みとは扱いません。[実測と範囲](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/m2-sample-cache-validation.md)を参照してください。
