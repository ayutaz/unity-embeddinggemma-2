# 必須端末の検証準備

更新: 2026-10-19。対象はmacOS EditorとiOS / Android Player実機です。Windowsの既存実測と区別し、設定テスト・Xcode export・実機build・実GPUの結果をそれぞれ記録します。

## iOSのRelease設定

`ValidationPlayerBuild.IOSOptions(output, sourceCommit)` は、固定40桁sourceと新しいXcode出力先を検査し、Device用のRelease exportを返します。既存exportや同じ親の以前のreceiptを上書きしません。`AutoRunPlayer` は指定しません。

`ValidationPlayerBuild.BuildIOS(output, sourceCommit)` は、iOSをactive targetにしたEditorで使用します。Play Mode・未保存scene・既存の一時検証assetがある場合は開始しません。既存の共通build経路がsource hash、実build結果、正常終了と設定復元を記録します。注入したbuild callbackは実export成功として記録されません。

profileはIL2CPP / High stripping / Release compiler、Device SDK、iPhone / iPad、Metalを指定します。[Xcodeの構成](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/EditorUserBuildSettings-iOSXcodeBuildConfig.html)も明示的にReleaseへ設定し、元のDebug / Releaseのどちらにも復元します。deployment targetの`15.0`は検証用buildの設定であり、実測済みの最低OS・端末要件ではありません。[Unity 6.3のiOS設定API](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/PlayerSettings.iOS.html)と、固定Editorでの設定テストを基準にしています。

`WithIOSProfile(Action)` は同じ設定を一時適用し、callback成功・例外の両方で変更した設定を復元します。自動signing、Team ID、provisioning値は読み取ったり変更したりせず、receiptには明示した設定項目だけを含めます。署名は実行者の私的なXcode環境で設定し、export・project settings・raw logは公開payloadとして扱いません。公開する結果は必ず別にprivacy検査します。

## 固定入力の端末内配置

`PlayerFixtureStager.Stage(sourceUri, newDirectory, manifestBytes, manifestSha256, report)` は、明示したlocal StreamingAssetsの入力を、新しい専用directoryへ取り出します。呼び出し側が渡すmanifestの元bytesとSHA256を先に照合します。schema 1の`files`配列には、各fixtureの`path`・`bytes`・`sha256`を含めます。相対pathの曖昧さ・重複・fileとdirectoryの衝突を拒否し、JSON / BIN / PNGだけを対象にします。上限は512ファイル、各64 MiB、合計512 MiBです。

転送はfile URI、Androidではjar:file URIを指定する前提です。Internetや共有サーバーからのモデル取得はこの処理の対象にしません。各`.part`のサイズ・SHA256を検証してから元の相対名へ移し、既存の出力先は使用しません。中断・転送失敗では処理中のpartialを除き、既に検証した入力と失敗記録を保持します。転送中のreportには例外型だけを記録し、元の例外messageや絶対pathを含めません。

stageの`success`は転送内容の照合だけを表します。`original_fixture_validated`・`gpu_verified`はfalseのままです。続いて既存の`PlayerMediaFixture.Load` / `PlayerMixedSearchFixture.LoadText`が独立した元hashを照合し、その後にモデル実行とGPU監査を行います。manifestの値を元の数値参照や許容差の代わりにしません。

[TDDと元入力の統合確認](results/device-fixture-materialization-tdd-20261019.json)は29件Red→29件Green、元28ファイル・34,474,958 bytesの実file URI転送と独立した全hash再計算が成功しました。変更していないloaderが画像・音声の全role 36ケースとtext 6文書・4 queryを受け入れ、元証拠と私的な完全2コピーを保持しています。Windows Editorでの転送・loader確認であり、Android jar転送・実機起動・CPU/GPU推論は未実行です。後続の[端末用runner](device-mixed-runtime.md)への接続とEditor契約検査は追加済みですが、実機による転送・起動の受け入れは残ります。

## 実機検証へ進むために残る作業

| 項目 | 現状 | 必要な証拠 |
| --- | --- | --- |
| iOS build profile | 設定・復元・上書き拒否のEditor契約を追加 | Macの固定Editor / Xcodeによる実export・署名・端末起動 |
| Android build profile | 既存のARM64 / IL2CPP / High / Release設定 | moduleを導入した実build、端末起動、実際に選択されたgraphics API |
| 画像・音声 / 混在Player入口 | Windows用単体入口の制限を維持し、別のAndroid / iOS混在入口を追加。関連190件のEditor契約検査が成功 | 実build・端末内設定入力・保存先・platform scopeを確認し、元数値gateを実機で実行 |
| 固定fixtureの配置 | pinned stagingと元28ファイルのfile URI / loader統合、端末用runnerへの接続を確認。Android jarは未実行 | Android StreamingAssetsからの実転送と元hash照合を実機で確認 |
| macOS / iOS / Android GPU | 未実行 | 全3入力・全role・混在索引、actual GPU / fallback / readback、独立CPU比較 |
| 端末の操作・寿命 | 未実行 | 権限許可 / 拒否、deviceなし、suspend / 復帰、解放、正常終了とメモリ |

AndroidのStreamingAssetsはfile APIで直接読めない場合があり、[Unityの公式手順](https://docs.unity.com/en-us/engine/6000.3/manual/building-and-publishing/streaming-assets)に従ってUnityWebRequest等で取り出す必要があります。転送完了だけを元fixtureのhash一致やGPU推論成功に数えません。

実機・署名・build環境が確保できるまで正式リリース条件は未達です。Windows向けの既存証拠は保持し、未実行・失敗・skipを実機合格へ置き換えません。

[Android / iOSの混在runtime入口](device-mixed-runtime.md)を追加しました。専用marker・端末内run設定・固定fixture転送・既存の全roleと混在GPU検査を接続し、関連190件のEditor検査が合格しました。実build・端末配置・GPUの受け入れは未実行です。
