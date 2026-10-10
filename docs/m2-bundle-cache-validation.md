# Player bundleのcache再利用と完全SHA-256高速化

URL展開をrunごとの新しいdirectoryへ行うと、反復mobile検証のたびに約1.68GBをcopyして保持してしまう。検証PlayerのURL経路を、`persistentDataPath/EmbeddingGemmaValidation/BundleCache`にある所有marker付きcacheへ変更した。配布UPM Runtimeは変更していない。

cacheは`active`の1 bundleを保持する。起動ごとにsourceの`bundle.json`を取得し、固定revision・source commit・Unity version・固定5ファイルとsize / hash宣言を検証する。manifestが一致していても、activeの全ファイルを完全SHA-256と固定参照で監査し直してから再利用する。sizeやmtimeだけで合格にはしない。

更新・破損修復では`pending`へ新しいbundleを作り、完全監査成功後にactiveへ切り替える。転送・監査失敗で旧activeを上書きせず、通常の更新中は最大2 bundle、成功後は1 bundleとする。中断した切り替えの`previous`を回復し、途中のpendingは次の試行時にdiscardする。所有marker・固定directory / file名・link guardの外にあるデータは削除せず、cleanupは再帰削除を使わない。

cacheのleaseは監査から全推論・worker解放まで保持する。完全監査から作ったassembly内部の証明をprotocolへ渡し、同じproduction runで全hashをもう一度計算する負担を避ける。明示的に注入したloader / tokenizer / providerは従来どおり`injected_contract`として扱い、実GPU成功を示さない。通常filesystem経路は同じ場所で監査する。

## TDDと失敗履歴

最終検証sourceは `b885b27cb9225291c9f46f9be2bdafc634cac8d9`。[ソースhash・NUnit・全数値](results/m2-bundle-cache-windows-20261010.json)を参照。

- cache新規10件の未実装redを観測し、起動経路のlease保持と破損manifest修復の追加2件でもredを確認した。
- 新しいactive出力先へ変わった既存テストの期待を更新し、cacheまでの全89件が合格した。整理で必要なLINQ参照を外したcompile失敗1件も記録し、参照を戻した。
- 実bundleのcache作成・warm再利用を測定したところ、全hash監査が各約108秒だった。両監査が終わりJSONは保存されたが、複合NUnitテストは180秒timeoutでfailedとなった。これをbaselineのテスト合格としては数えない。
- 完全hashの新規7件でredを観測し、Windows CNG、既知のSHA-256値、5MB超の独立実装との一致、prefix / 最終partial block、native未使用・不正出力時の全bytes fallback、IO失敗・無効streamの拒否を確認した。既存を含む96 passed / failed・skipped・inconclusive 0だった。

## 同じ実bundleのWindows Editor測定

Unity 6000.3.16f1 / Sentis 2.6.1 / Windows 11の同じEditorで、既存prepared weightsを使った。model revisionと全5ファイルのSHA-256は以前の監査済みbundleのまま。モデルの再download・変換はしていない。

| warm cache監査 | 完全hash・参照parse ms | 全体 ms | model再転送 |
| --- | ---: | ---: | ---: |
| baselineの観測（NUnit全体はtimeout失敗） | 108761.4 | 108773.5 | 0 |
| CNG 1 | 1502.7 | 1524.5 | 0 |
| CNG 2 | 1464.9 | 1471.2 | 0 |
| CNG 3 | 1752.4 | 1759.1 | 0 |

高速化後の測定テストは1 passed / failed・skipped・inconclusive 0。全5ファイルで実際に`windows_cng`を使用したことを記録し、毎回manifest 1件だけを転送した。保持bundleは1 directory / 1682221728 bytes。baselineのcache作成時は6件転送、109111.2msで、107462.0msが監査だった。

同じcache・既存file / OS cacheでbaselineのwarm観測1回とCNGのwarm観測3回を比較したもので、cold-machine・統計的benchmark・Player起動全体の速度倍率ではない。全体時間は転送・完全SHA-256・固定参照parse・directory操作を含む。Sentis推論 / GPU精度 / peak memoryの新しい実行結果ではない。

## hash実装と再現

`PlayerFileHash`はcallerが開いている同じseekable streamの先頭から全bytesをhashする。Windows compile targetではOSの[BCryptCreateHash](https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptcreatehash) / [BCryptHashData](https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcrypthashdata) / [BCryptFinishHash](https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptfinishhash)を使い、1MiB bufferで全体を処理してhandleをfinallyで閉じる。native実装が利用できない場合・他OSは、streamを先頭へ戻して.NET SHA-256で全bytesを照合する。各ファイルの実際の`hashBackends`をstageとPlayer結果へ保存する。

uloop-cliで`EmbeddingGemma.Player.Validation.Tests`のEditModeを実行すると、weights不要の96契約を検証できる。実bundle測定はローカルの独立したmeasurement assemblyを使い、そのソースhashを結果へ保存した。再現時は同じ監査済みbundleを`PlayerBundleCache.Resolve(fileUri, cacheRoot, stage)`へ渡し、source hash・OS / Unity・manifest・件数・`auditPassed`・`cacheReused`・backend・時間を保存する。新しいcache rootで作成を測り、同じrootでwarmを測る。以前の証跡JSONを上書きしない。

実行後はEditor 1つ、Play停止・scene変更なし、ConsoleにはTest Runnerの通常Log 3件。model / 大きいcache / ローカルfixtureはGitに含めない。

## 残る条件

Windows Player / IL2CPP / release stripping、Android APK内のjar・実機のlock / rename / 容量・CPU / GPU、macOS Editor / iOSは未実行。Windows CNGのEditor合格でPlayer・他OSの成功を代替しない。native fallbackの精度は小さいデータで確認したが、他OSの実model性能は未測定。

配布sampleへAndroidの配置経路を渡す作業、allocation / domain reloadの原因と正式リリースのgateも残す。途中転送のbyte-range resumeは未実装。小さいrun別reportは別途保持し、以前の`StagedBundle-<run ID>`等の未知のdirectoryを自動で採用・削除しない。
