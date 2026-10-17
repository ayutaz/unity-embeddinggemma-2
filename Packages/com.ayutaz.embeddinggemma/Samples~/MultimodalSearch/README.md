# 画像・音声検索サンプル

Package Manager の EmbeddingGemma の Samples から **Image and Audio Search** を Import し、`MultimodalSearch.unity` を開きます。モデルはパッケージに含まれません。

1. 監査済みの `bundle.json` と `objects/` を含む準備済みモデルフォルダーを指定します。画像・音声の登録用moduleと、検索用の `text` moduleが必要です。
2. PNG / JPEG画像、またはmono 16 kHz音声ファイルの絶対パスを指定します。WAVはPCM16 / IEEE float32、`.f32`はlittle-endian float32に対応します。
3. 対応する固定profileを選び、**画像・音声を登録**を押します。別のファイルを続けて登録できます。
4. 検索文を入力し、**検索**を押します。画像・音声を一緒に順位付けし、score・ファイル名・パスを表示します。

画像は前処理後の寸法が square=768×768、landscape=1008×576、portrait=576×1008 のprofileを使用します。音声は1119 / 839 valid frameの固定profileです。音声は161〜179200 sampleを受け付けますが、現在準備できるencoderのframe数に一致する必要があります。入力の暗黙のresampling・切り詰め・clippingは行いません。

モデルのSHA256とサイズを照合し、選択moduleに必要なファイルだけを `Application.persistentDataPath/EmbeddingGemmaMediaCache` へ保存します。次回は正しいファイルを再利用し、破損したファイルだけを再取得します。別moduleへの切替では前のGPU処理完了とモデル解放を待ちます。登録済み768次元vectorは保持します。ファイル名やパスをテキスト文書として埋め込むことはありません。

**キャンセル**は読み込み中、またはencoderのGPU処理境界で反映されます。失敗・キャンセル後は同じ画面から再実行できます。GPUComputeに非対応の環境やCPUへの計算fallbackが必要なモデルは拒否します。

このサンプルはWindows Editor向けです。他OS・モバイル・WebGLの動作確認は別途必要です。準備処理の正常終了と、実GPUでの数値検証は別の結果として扱います。モデル準備の固定source・元参照・実行条件はrepositoryの検証記録を参照してください。

**新規利用者向けの制約**：モデルbundleの公開配布と、空のcheckoutからの一括生成手順は未整備です。開発環境に存在するローカルフォルダーはパッケージには含まれません。[準備経路と未整備の範囲](https://github.com/ayutaz/unity-embeddinggemma-2/blob/main/docs/model-preparation.md)を確認してください。モデル・係数・tokenizerはGitへ含めません。
