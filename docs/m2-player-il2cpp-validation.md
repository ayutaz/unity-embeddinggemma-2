# Windows IL2CPP release Playerの検証

2026-10-10。Windows Playerの既存Development / Mono検証に続き、IL2CPP / Release / managed stripping Highで同じ固定実モデル・Python参照を検証する。

## build helperとTDD

`ValidationPlayerBuild.BuildWindows`に`releaseIl2Cpp: true`を追加した。指定時は`BuildOptions.None`、StandaloneのIL2CPP、High stripping、IL2CPP compiler Releaseを選ぶ。実際のbuild receiptにoptionsとcompiler設定を記録し、元のbackend・stripping・compiler設定とsceneを成功・失敗のどちらでも復元する。既存の呼び方はDevelopmentのまま。

新規3ケースの意図したred（既存5 passed / 新規3 failed）を確認してから実装し、既存Player / cache / hash契約を含む99 passed / failed・skipped・inconclusive 0を確認した。release options、実行時の設定、成功・失敗の設定復元、注入buildを実build成功と扱わない条件を検証した。この合成テストの合格は実モデル・GPUの合格とは別。

```csharp
ValidationPlayerBuild.BuildWindows(
    "<new empty directory>/Validation.exe",
    "<40-character source commit>",
    releaseIl2Cpp: true);
```

Unity 6000.3.16f1 / Sentis 2.6.1 / Visual Studio 2026のWindows toolchainを使用した。source checkpointは`c742fb940b78393573b19e42352fe7b70256e74d`。同じ1つのEditorでbuildを1回受理し、257.1328秒、errors 0 / warnings 485でSucceededとなった。Sentis shaderの警告を含み、警告ゼロの結果ではない。

受理後にwindow focus操作がtimeoutしたが、buildは同じEditorで進行し、完了receiptを確認した。focus timeoutをbuild失敗にせず、build再投入・Editor再起動をしなかった。build後は元のsample scene、Play停止・scene clean、Mono2x / stripping Disabled / compiler Releaseへの復元を確認した。

## 実行の範囲

検証専用assemblyには従来の`link.xml`でpreserve-allを指定している。配布UPMにこのassemblyを含めず、すべてのconsumerの任意のstripping設定へ結果を一般化しない。

consumerは当初固定Git SHA `e7b3a54`を解決した`artifacts/h`で、Core Runtimeは検証sourceとの一致を確認した。import済みvalidationを記録したsourceへ置き換えた条件であり、`c742fb9`の新規Git導入成功とは扱わない。以前のprepared weightsとimport / shader / filesystem cacheを再利用し、モデルの再download・変換はしていない。

Windows 11 / RTX 4070 Ti SUPER / Direct3D12で同じbinaryを2回、異なるrun ID・結果先で起動した。前のprocess終了後だけ次を開始した。固定bundleを`file://`で渡し、初回の6ファイル展開と、2回目のmanifestのみ転送・全5ファイル完全hash・cache再利用を確認した。実際のCPU / GPUCompute × FP32 / Float16重み、25入力token ID / mask一致、M1 15ケース、検索6文書 / 4queryの全順位・score・解放を機械監査した。

## 実Playerの結果

[source・TDD・build・2起動の機械監査結果](results/m2-player-il2cpp-windows-20261010.json)。両方とも終了code 0、`real_model` / `isEditor=false`、4条件のactual backend一致、全25入力のtoken ID / mask、有限768次元・単位vector・M1 / 検索のcosineと全順位・score・worker解放が合格した。最低cosineは`0.999999999998`以上、最大score差は`2.43e-7`未満。

| 確認 | 初回起動 | cache再利用起動 |
| --- | ---: | ---: |
| process起動から終了まで 秒 | 83.5530 | 78.5791 |
| bundle展開・監査 全体 ms | 3338.1 | 1573.3 |
| 完全SHA-256・参照parse ms | 1474.5 | 1554.3 |
| 転送ファイル数 | 6 | 1（manifestのみ） |
| モデル再転送 bytes | 初回配置 | 0 |

全5ファイルのhash backendは実際に`windows_cng`だった。監査済みcertificateで二重hashを避け、cache leaseを推論・worker解放まで保持した実行経路で合格した。同じcacheに成功後の`active`だけを保持し、6ファイル / 1682221728 bytesを確認した。

| 重み / backend | warmup後query平均 ms: 起動1 / 2 |
| --- | ---: |
| FP32 CPU | 514.1 / 519.1 |
| FP32 GPUCompute | 44.0 / 33.8 |
| Float16 CPU | 832.9 / 886.7 |
| Float16 GPUCompute | 34.5 / 35.5 |

各平均は同じqueryの3回。FP16は重みprecisionで、すべての演算がFloat16という主張ではない。モデルdeserialize、worker / tokenizer準備、初回推論、warmup後の3回、解放とapplication全体のメモリstage sampleは結果JSONに保持した。Release Player logに以前のDevelopment Playerの`MemoryLeaks`行はなく、結果の診断値はnullとした。診断行がないことをnative leak解消の証拠にしない。完了後は検証Playerなし、Editor本体1つを確認した。

## 残るgate

Windowsのfile URLはAndroid APK内のjar読み込みの証拠ではない。macOS Editor / iOS / Androidの実機CPU / GPU、sampleのdevice読み込み、allocation / domain reloadの追跡、正式リリースgateは引き続き未達。時間は既存cacheを利用した反復実行で、cold-machineの比較や統計的benchmarkではない。メモリはapplication全体のstage sampleで、peak・モデル専用・GPU使用量の証拠ではない。
