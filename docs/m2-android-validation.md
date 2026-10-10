# Android検証APKのbuild・監査と実機gate

確認日: 2026-10-10。実装開始の基準mainはPR #25後の `b1a8c67c62066bfa5e1830a947563ace3dff9fa5`。その後PR #27の文書更新main `c3281cd`を取り込んだ。Runtime / sample / GUID / manifestは不変で、統合後CI run 38025309104は全8 job成功。
source `751c30a0b7216b4fe090e989118e570e06b10775`の実APK build・SDK署名検証・完全payload監査は成功した。インストール・実機CPU / GPUは未実行。
[実APK結果](results/m2-android-apk-build-20261010.json)、[準備時点の履歴](results/m2-android-preparation-20261010.json)、[4段階の計画](m2-release-plan.md)を参照。

## 実装

`ValidationPlayerBuild.BuildAndroid`は停止中・scene保存済み・active target Androidを要求する。
固定40桁commitと空の出力ディレクトリを指定し、生成した検証sceneだけを使う。
Windowsと同じreceipt / Runtime source hash / build marker / scene復元を共有し、Androidの設定を一時適用する。

- ARM64 / IL2CPP / compiler Release / High stripping。
- application identifier `com.ayutaz.embeddinggemma.validation`、min SDK 26、target SDK Auto。
- graphics APIはVulkan、OpenGLES3の順。実行時のactual APIはPlayer結果から別途確認する。
- APK単体。AAB、OBB、ABI別APK、Gradle project exportを無効にする。
- custom keystoreを無効にして検証APKを作る。秘密鍵・パスワードの値は読み取らない。
- Developmentが既定。実機から検証結果を回収するための検証用buildであり、製品の配布設定ではない。
- 元のbackend / compiler / stripping / ABI / identifier / SDK / packaging / graphics設定を成功・失敗とも復元する。

automatic graphics APIが有効な状態では`GetGraphicsAPIs`が既定リストを返すため、保存済みの明示リストを復元する契約が最初に失敗した。
snapshot中だけautomaticを無効にして保存済みリストを読み、直ちにautomaticを戻す実装で修正した。

`tools/embeddinggemma_tools/android.py`はbuild receiptとAPK内の全5 payloadの長さ・完全SHA-256、bundle manifest、ARM64 native librariesを監査する。
ZIPの重複entry、余分なモデル、他ABI、injectしたbuild receipt、source commit不一致を拒否する。
署名、インストール、Androidのjar読み込み、推論精度、GPU実行はこの監査の範囲外で、成功時も`unity_runtime_executed` / `gpu_verified`はfalseを維持する。

## 確認済みのテストと限界

| 実行 | 結果と対象 |
| --- | --- |
| Android helper初期red | Windows targetの同じEditorで9 failed / 0 passed。未実装stubの意図した失敗 |
| 最初のgreen試行 | 108件中106 passed / 2 failed。graphics復元の確認方法を見直したため、合格扱いしない |
| graphics保存リストのred | 11件中9 passed / 2 failed。automatic=trueの成功・失敗とも保存済みOpenGLES3リストを復元できなかった |
| 修正後Windows target契約 | 110 passed / failed・skipped・inconclusive 0。既存99件とAndroid準備11件。compileも成功 |
| APK監査red | 小さいZIP fixtureの16 failed。未実装stubによる意図した失敗 |
| APK監査green | 新規16件と既存bundle13件、29 passed。実APKを使った結果ではない |

**110件の合格後**に旧`useAPKExpansionFiles`をUnity 6.3の`splitApplicationBinary`へ置き換え、Android targetでreceiptとscene/settings復元を検証する2ケースを追加した。
その変更を含む現在のC#はAndroid targetでcompile成功、契約12 passed / failed・skipped・inconclusive 0を確認した。scene / settings復元の2ケースも別に2 passedを確認し、12件に含まれるため合計14件とは数えない。Windows用target拒否ケースは今回の対象外。110 passedは変更前の履歴として維持する。
target切り替えを1回だけ受理した後、同じEditor PID 110956のCLI観測は応答待ちとなった。
ウィンドウタイトルはAndroidになったが、`Editor.log`はAPI Updaterの出力で止まっており、原因は未確定。
同じ観測CLIは約30分後に`UNITY_RESPONSE_TIMEOUT_AFTER_ACCEPT`で終了し、`SafeToRetry=false`を返した。
Editor本体は生存し、uloop statusは`MainThreadBlocked` / compilingと報告した。CLIの終了はEditorや受理された操作の終了証拠ではない。
約37分の継続したMainThreadBlocked、実質的に進まないログとCPU、保存済みsceneとcommit済みsourceを確認した後、旧Editorとその子workerを終了した。旧PID終了・Editor本体0を確認してから、Android targetのEditorを1回起動した。新PIDは127008。起動・compile・12契約・実buildが成功した。観測timeoutだけを終了証拠にして再起動したものではなく、停止原因が解決・特定できたとも扱わない。
後のプロセス確認でも画面を持つEditor本体は1つ。AssetImportWorker 2つは同じ本体が起動した補助プロセスだった。

## 実APKの確認結果

- Unity 6000.3.16f1 / Sentis 2.6.1、ARM64 / IL2CPP / compiler Release / High strippingのDevelopment APK。実buildは504.158秒、errors 0 / warnings 970。build開始05:00:37 UTC、完了05:09:02 UTC。
- APKは1,720,031,692 bytes、SHA-256 `efcde071bc540ac093482ea7ce340eb6904d2d6e9ab012c5174d3210edba1ca1`。BuildReportのサイズ3,773,032,313 bytesとは別の実APK長を記録する。
- SDK build-tools 36.0.0のapksigner検証exit 0、v2署名成功。Android Debug signerの検証であり、製品Release署名ではない。v1 / v3 / v4 / SourceStampを成功とは扱わない。
- aapt2 exit 0。min SDK 26 / target・compile SDK 36、`arm64-v8a`のみ、launcherは`com.unity3d.player.UnityPlayerGameActivity`。アプリversionName 1.0は検証アプリの設定で、UPM versionとは別。
- bundle.jsonと全5 payloadの長さ・完全SHA-256が固定bundleと一致し、libunity / libil2cpp、余分なモデルや他ABIがないことを確認した。Runtime 14ファイルのLF正規化hashをbuild sourceと照合した。consumerのCoreはGit `01f3d86`由来で、今回のsourceと一致する。
- sampleモデルは所有project内のAssets外へ退避し、既存bundleの6ファイルをhardlinkして二重packagingと再download / conversionを避けた。build後はsampleとmetaを復元し、検証payloadをAssets外に保持した。生成scene / markerは除去済み。元sceneは保存済み・Play停止・compile停止、strippingは元のMinimalに戻った。
- 警告の例はSentis Pad / ConvTranspose shaderの整数剰余演算の遅さ（gles3 / vulkan）。全970件の原因分類・解消や端末性能の改善は未確認。

実receiptのBuildOptionsは`ForceOptimizeScriptCompilation, Il2CPP, CompressTextures, StripDebugSymbols, ShaderLivelinkSupport, Development`だった。APK監査はこの固定Unityの完全な文字列表現だけを追加許可した。1件の意図したredと2件の拒否成功を観測して修正し、APK契約19 passedを確認した。任意の追加flag、AutoRunPlayer、IncludeTestAssembliesを許可する変更ではない。
APK監査結果のruntime / GPUフラグはfalseのままである。

## build再現と残る実機手順

下記1〜6は今回確認済み。現在は7の接続実機待ちで、不要なtarget切り替えやAPK再buildはしない。

1. 既存Editorの応答を確認する。先に受理された観測を追跡し、同じtarget切り替えを二重発行しない。
2. 現在のC#をcompileする。`ValidationAndroidBuildTests.BuildRecordsAndroidProvenanceAndRestoresTheSceneAndSettings`の2ケースを明示的に実行する。
   `WrongActivePlatformCannotGenerateOrModifyAssets`はAndroid以外のtarget用であり、同時に実行しない。
   Windows build契約はWindows targetで確認する。target固有のfixtureを全件実行して対象違いの失敗を混ぜない。
3. 検証に使うsource commitを固定し、consumerへ反映したRuntimeとvalidation sourceを照合する。
4. 検証consumerのTextSearch用StreamingAssetsモデルを、パスの所有範囲と退避先を確認して退避する。
   `Assets/StreamingAssets/EmbeddingGemmaValidation/`へ監査済みbundle.jsonと5ファイルだけを配置する。
   既存bundleを再利用し、追加download / conversionやsampleモデルとの二重packagingを避ける。
5. 空のbuild出力を指定して`ValidationPlayerBuild.BuildAndroid`を1回だけ実行する。build.jsonが完了するまで同じEditor / buildを追跡する。
6. 実build完了後、署名検証とAPK監査を実施する。Pythonはtoolsでuvを使う。

```powershell
uv run --locked python -m embeddinggemma_tools.android `
  --apk <Validation.apk> --bundle <bundle.json> --build <build.json> `
  --commit <40桁のbuild source SHA> --output <新しいapk-audit.json>
```

7. ADBで接続端末を確定してインストール・独立2起動を実施し、Player結果を回収する。
   初回jar展開 / warm cache / 完全hash / 再起動 / 解放 / requestedとactual backendを確認する。
   固定token・M1 embedding・検索全順位・scoreをFP32 / Float16 × CPU / GPUで照合し、速度・メモリも記録する。
8. 配布sampleのAndroid Player画面も別に操作確認する。validation用APKの成功でsample UI成功を代用しない。

Android module / SDK / NDK / JDKは利用可能だが、確認時ADB接続端末は0件。
実APK・署名 / payload監査は確認済み。実機、macOS / iOS、未解決のallocation / font / domain reload、tag導入、正式Releaseは開いたgateとして維持する。
PRのC#再検証待ちは解消した。最新headの通常CIと差分を確認してハーネス変更を統合し、実機合格とM2完了は引き続き別に判断する。

## 実機結果の回収と報告監査

監査済み実APKは確保したが、以下はまだ接続実機で実行していない。
[Android公式ADB手順](https://developer.android.com/tools/adb)に従い、serialを明示して対象を固定する。
APK内のlauncher componentを確認してから起動し、Windows用のcommand line引数や固定のActivity名を流用しない。
bootstrapは新しいGUIDのdirectoryへresults.jsonを保存し、別の32桁run IDも生成する。
Androidの保存先は通常`/storage/emulated/<userid>/Android/data/<packagename>/files`だが、
[Unity 6.3のpersistentDataPath仕様](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Application-persistentDataPath.html)と実際のPlayer出力を照合し、useridや保存先を仮定しない。

```powershell
adb devices -l
adb -s <serial> install <Validation.apk>
adb -s <serial> shell am start -W -n com.ayutaz.embeddinggemma.validation/com.unity3d.player.UnityPlayerGameActivity
adb -s <serial> shell pidof com.ayutaz.embeddinggemma.validation
adb -s <serial> pull <実際のGUID-directory/results.json> <新しいrun1.json>
```

初回のPID生存、結果completed、PID消失をそれぞれ時刻付きで保存する。
結果completedだけで終了済みとは扱わず、同じ実行を追跡する。停止・失敗の結果も上書きしない。
1回目の終了確認後、同じAPKで2回目を起動して新しいPID / run ID / results.jsonを回収する。
キャッシュを保ってwarmを測る。モデルを含むBundleCache directory全体をpullせず、各results.jsonと必要な検証ログだけを回収する。
端末serialとOS / GPU / API / ABIの対応、2つの起動と終了の観測は別のADB証拠へ保存する。

`android_runs`は回収された2つの報告の整合性を監査するCLIで、ADBを実行しない。
固定APK auditと同一build source hashes / bundle、同梱参照の完全hash、tokenRows 25、
FP32 / Float16 × CPU / GPUComputeの全4条件、actual backend、M1全15ケースのcosine、
検索全4queryの全6順位・score、worker解放、3回warm timing、14段階のメモリ観測、
初回jar全6件展開から2回目manifest 1件のwarm cache移行を要求する。
cosine・scoreは既存のPlayer条件と同じ閾値を使う。
同じrun ID、期間重複、途中報告、injected結果・hash backend、CPU fallbackやWindows / Editor結果は拒否する。

```powershell
uv run --locked python -m embeddinggemma_tools.android_runs `
  --runs <run1.json> <run2.json> --apk-audit <apk-audit.json> `
  --reference <reference.json> --search-reference <search-reference.json> `
  --output <新しいplayer-report-audit.json>
```

監査成功の`reported_gpu_pass`はPlayerが報告した結果の整合性を表す。
このCLIはembedding / tokenを再計算せず、端末identityや独立process終了も証明しない。
`android_execution_performed_by_auditor` / `independent_process_exits_verified`はfalseを維持する。
実機合格には上記の実APK・署名 / payload・起動 / 終了・端末情報の証拠と実際のPlayer結果を組み合わせる。

報告監査は39件の意図したredと、追加5件のredを観測して実装した。
追加のredはinjected hash、欠けたmemory stage、重複JSON field、NaNと数値overflowの拒否を確認した。
新規44件と既存APK / bundle 29件の計73 passedを確認した。小さいfixtureのみで、実Android結果はまだない。
