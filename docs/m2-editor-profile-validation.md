# Editor待ち時間の切り分け: 2026-10-12

Unity 6000.3.16f1の既存Editorと保存ログを調べ、[小さい公開要約](results/m2-editor-profile-windows-20261012.json)へ記録した。追加Editor起動、compile、Play、モデル取得、buildは行っていない。これは計測手段と履歴の分析であり、domain reload高速化や終了クラッシュ修正の完了ではない。

## 計測ツール

`tools/embeddinggemma_tools/editor_profile.py`をTDDで追加した。未実装で16 failedを確認してから実装し、16 passedを確認した。完了済みreload、未完了のbegin、beginが欠けたprofile、内訳、asset refresh、SearchDatabase列挙例外とPersistent allocation診断を分ける。生ログを公開せず、既知のprofiler名・数値・snapshot SHA-256だけを出力する。

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools.editor_profile --log <saved-Editor.log>
```

CLI exit 0は、保存ログに完了profileがあり、未対応のbegin / profileがないことだけを意味する。Unityの正常動作、実モデル精度、エラー解消、性能改善やM2合格を意味しない。未完了やprofile欠落はexit 1で、停止時間の推定や自動再起動は行わない。現在のPID・応答や進行中の処理は別途確認する。

## 観測結果

| 保存ログ | 完了reload数 | min / median / max | Asset refresh数 / median / max | SearchDatabase例外 / allocation診断 |
| --- | --- | --- | --- | --- |
| jの既存session | 30 | 0.913 / 3.866 / 7.939秒 | 226 / 0.041 / 65.071秒 | 4 / 1 |
| iの過去の復旧前snapshot | 16 | 0.979 / 3.828 / 7.984秒 | 193 / 0.035 / 80.624秒 | 1 / 2 |

両snapshotに未対応のreload begin / profileは0件。iのログはtarget変更後の通信停止・復旧前に保存されたが、完了したreloadの記録だけで当時の応答性や停止原因を判断できない。以前の別consumerでの数分間のdomain reload停止も、この2つの統計だけでは説明できない。

jの最大65.071秒のrefreshは初回の3015 actual imports / local cache 0件で、Asset DB native process 27.908秒、native callback 17.241秒、compile 16.427秒が記録されている。同じrefreshのScripting欄のdomain reload timeは2.300秒。独立したDomain Reload Profilingとは測定境界が違うため同一値として比較しない。初回待ち全体をdomain reloadだけの問題として扱う根拠はない。

jの最後の完了profileは3.865秒で、FinalizeReload 2.559秒、SetupLoadedEditorAssemblies 1.252秒、ProcessInitializeOnLoadMethodAttributes 0.672秒、AwakeInstancesAfterBackupRestoration 0.667秒。これらは包含関係のある時間なので合計しない。session内に起動・コード変更・Play/test・buildと後続source配置が混在するため、固定candidateの性能基準にも使わない。

現在のuloop Console Errorは0件、Editor本体は1件、子のAssetImportWorkerは2件だった。過去ログのSearchDatabase列挙例外4件とallocation診断1件は保持する。ADB接続端末・Repository Secrets・self-hosted runnerは各0件だった。

## 次の条件

1. 初回asset import / compile、完了reloadの内訳、target変更・通信停止を別々に追跡する。モデル・Library・build cacheを維持し、ログ採取のために全importや再buildを誘発しない。
2. 最適化対象を変更する前に、同じEditor / source / backendの対照条件を決める。既存cacheを使った値をcold start改善と表現しない。
3. Player終了クラッシュの修正版比較は固定Unity版変更の回答後に行う。独立2起動・正常終了、Android / Mac / iOSの実測、正式candidate / Releaseのgateを維持する。
