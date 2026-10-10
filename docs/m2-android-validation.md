# Android検証の準備と未実行gate

確認日: 2026-10-10。基準mainはPR #25後の `b1a8c67c62066bfa5e1830a947563ace3dff9fa5`。
これはAndroid向け検証ハーネスの準備記録であり、APK build / インストール / 実機CPU・GPUの合格記録ではない。
[機械可読結果](results/m2-android-preparation-20261010.json)と[4段階の計画](m2-release-plan.md)を参照。

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
その変更を含む現在のC#は再compile / 再テスト待ちである。110 passedを現在のC#全体の合格証拠にしない。
target切り替えを1回だけ受理した後、同じEditor PID 110956のCLI観測が応答待ちになっている。
ウィンドウタイトルはAndroidになったが、`Editor.log`はAPI Updaterの出力で止まっており、原因は未確定。
確認ダイアログの有無を確認中で、観測timeoutを理由に再起動・target再要求はしていない。
Editor本体1つと、その子AssetImportWorker 2つを確認した。追加のEditor本体は起動していない。

## 再開手順

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
APK build自体はまだ要求しておらず、実APK、実機、macOS / iOS、未解決のallocation / font、tag導入、正式Releaseはすべて開いたgateとして維持する。
現在のPRはC#再検証待ちのdraftとし、通常のActions CI成功だけでmergeしない。

## 実機結果の回収と報告監査

実APK / 接続端末を確保した後に使う手順で、以下はまだ実機で実行していない。
[Android公式ADB手順](https://developer.android.com/tools/adb)に従い、serialを明示して対象を固定する。
APK内のlauncher componentを確認してから起動し、Windows用のcommand line引数や固定のActivity名を流用しない。
bootstrapは新しいGUIDのdirectoryへresults.jsonを保存し、別の32桁run IDも生成する。
Androidの保存先は通常`/storage/emulated/<userid>/Android/data/<packagename>/files`だが、
[Unity 6.3のpersistentDataPath仕様](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Application-persistentDataPath.html)と実際のPlayer出力を照合し、useridや保存先を仮定しない。

```powershell
adb devices -l
adb -s <serial> install <Validation.apk>
adb -s <serial> shell am start -W -n <確認したpackage/launcher-component>
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
