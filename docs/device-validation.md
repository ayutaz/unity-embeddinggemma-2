# 必須端末の検証準備

更新: 2026-10-19。対象はmacOS EditorとiOS / Android Player実機です。Windowsの既存実測と区別し、設定テスト・Xcode export・実機build・実GPUの結果をそれぞれ記録します。

## iOSのRelease設定

`ValidationPlayerBuild.IOSOptions(output, sourceCommit)` は、固定40桁sourceと新しいXcode出力先を検査し、Device用のRelease exportを返します。既存exportや同じ親の以前のreceiptを上書きしません。`AutoRunPlayer` は指定しません。

`ValidationPlayerBuild.BuildIOS(output, sourceCommit)` は、iOSをactive targetにしたEditorで使用します。Play Mode・未保存scene・既存の一時検証assetがある場合は開始しません。既存の共通build経路がsource hash、実build結果、正常終了と設定復元を記録します。注入したbuild callbackは実export成功として記録されません。

profileはIL2CPP / High stripping / Release compiler、Device SDK、iPhone / iPad、Metalを指定します。[Xcodeの構成](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/EditorUserBuildSettings-iOSXcodeBuildConfig.html)も明示的にReleaseへ設定し、元のDebug / Releaseのどちらにも復元します。deployment targetの`15.0`は検証用buildの設定であり、実測済みの最低OS・端末要件ではありません。[Unity 6.3のiOS設定API](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/PlayerSettings.iOS.html)と、固定Editorでの設定テストを基準にしています。

`WithIOSProfile(Action)` は同じ設定を一時適用し、callback成功・例外の両方で変更した設定を復元します。自動signing、Team ID、provisioning値は読み取ったり変更したりせず、receiptには明示した設定項目だけを含めます。署名は実行者の私的なXcode環境で設定し、export・project settings・raw logは公開payloadとして扱いません。公開する結果は必ず別にprivacy検査します。

## 実機検証へ進むために残る作業

| 項目 | 現状 | 必要な証拠 |
| --- | --- | --- |
| iOS build profile | 設定・復元・上書き拒否のEditor契約を追加 | Macの固定Editor / Xcodeによる実export・署名・端末起動 |
| Android build profile | 既存のARM64 / IL2CPP / High / Release設定 | moduleを導入した実build、端末起動、実際に選択されたgraphics API |
| 画像・音声 / 混在Player入口 | 既存runnerはWindows Player限定。制限を緩めて実機合格にはしない | 明示的な端末用設定入力、保存先、platform別scopeと同じ数値gate |
| 固定fixtureの配置 | 既存loaderは通常のfile pathを要求 | AndroidのStreamingAssetsから端末専用領域へ取り出し、元hashを照合 |
| macOS / iOS / Android GPU | 未実行 | 全3入力・全role・混在索引、actual GPU / fallback / readback、独立CPU比較 |
| 端末の操作・寿命 | 未実行 | 権限許可 / 拒否、deviceなし、suspend / 復帰、解放、正常終了とメモリ |

AndroidのStreamingAssetsはfile APIで直接読めない場合があり、[Unityの公式手順](https://docs.unity.com/en-us/engine/6000.3/manual/building-and-publishing/streaming-assets)に従ってUnityWebRequest等で取り出す必要があります。転送完了だけを元fixtureのhash一致やGPU推論成功に数えません。

実機・署名・build環境が確保できるまで正式リリース条件は未達です。Windows向けの既存証拠は保持し、未実行・失敗・skipを実機合格へ置き換えません。
