# M3実画像encoderのSentis受け入れハーネス

更新日: 2026-10-13。[画像計画](m3-plan.md)のI2に対するEditor受け入れを追加した。**最終768次元、GPU画像前処理、画像API / 検索sample、他OS / 実機は未検証**。正式Release / tagは作成しない。

## 固定入力と実行条件

- [PR #37実モデルCI 38310718250](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38310718250)のartifact。reference sourceはCI merge ref `e2a6dc01fbca8c9be1f3bbe875325ad338de0ca0`、model revisionは`914f7f89142e33e77833254d9c9b90c3cef7303b`。checkout headとartifact sourceを混同しない。
- 完了stage、3 profile / 6入力、全graph / binary / 参照JSONのsize / SHA-256とsourceを監査する。tinyモデルでは通らない実入力shapeとfeature幅512を要求する。
- Windows / Unity 6000.3.21f1 / Sentis 2.6.1、RTX 4070 Ti SUPER / Direct3D12。固定Git consumer `artifacts/k`、RuntimeはPR #31の固定導入source。公式Test Framework 1.6.0のembedded条件を使用する。新しい画像test fixtureをconsumerのAssetsへ配置した実行であり、新candidateの空Git導入・16f1・他OSの成功ではない。
- square / landscape / portrait × FP32 / Float16 × CPU / GPUComputeの12条件。各条件で独立Pythonの2画像 / 全featureを照合し、`.sentis`保存・再読込、actual worker / output backend、finite / shape / normを検査する。
- cosineはFP32 >= 0.999 / Float16 >= 0.99。FP32はPython export gateと同じelementwise rtol `1e-4` / atol `1e-5`も検査し、超過を失敗として残す。診断featureのCPU取得はこのテストの測定境界であり、production GPU chainの成功とはしない。

## 初回実行の結果

[初回12条件の数値・失敗](results/m3-vision-windows-initial-20261013.json)。2026-10-12 22:10:50〜22:33:35 UTCに同じEditorで実完了した。CLIはaccepted後のmain-thread観測で`UNITY_EDITOR_UNRESPONSIVE` / `SafeToRetry=false`となったが、実行を重複させず、同じreportの12条件とcompletionを回収した。

| 条件 | 結果と限界 |
| --- | --- |
| FP32 / CPU 3 profile | cosine最小 `0.9999999992175514`。最大絶対誤差 `0.00018829107284545898`、elementwise許容差比7.14〜8.99で失敗。cosineだけで全合格としない |
| Float16 / CPU 3 profile | 全feature cosine >= 0.99で合格。最小 `0.9999999991103836`。この重み形式の精度のみで、速度改善の証明ではない |
| GPU 6条件 | 最初の監査がCPU定数を計算fallbackと誤判定し、数値推論前に失敗。実GPUの数値はこの初回結果にはない |

初回fixture SHA-256は`ac4a32eab083e343dc430c267b5aa695164449ffe388ea7e5313719b73d5cc70`。コンパイル時のJSON assembly参照・非対応NUnit属性・配列取得APIの失敗は修正し、compile Error 0 / Warning 0後に実行した。これらをアルゴリズムTDDのredへ加算しない。

## 監査修正と準備の最適化

Sentis 2.6.1の`m_LayerCPUFallback`は、CPUに置く定数のtensor indexも含む。保存したsquare / FP32 graphのschema v8 descriptionを読み、997件すべてがconstant、CPUで生成されるlayer outputは0件と確認した。hash set全体を空にする初回assertionはGPU計算の判定として誤りである。

実modelのlayer outputとmodel inputのindexに照合し、CPUで計算するtensor・CPU dataが必要な入力を拒否する監査へ修正した。`m_LayerCPUFallbackShouldFlushGPU`も空であることを要求し、中間GPU→CPU転送を許可しない。CPU定数・実CPU計算・入力CPU要求の小さい契約で2件の意図したredを確認した。

同じgraphのimport / 保存をCPU / GPU条件ごとに繰り返さず、profileごとにimportを1回、profile / precisionごとに保存を1回とし、各backendは同じ保存済みモデルを独立にreloadする。fixture開始時にcacheを消去し、前回runの準備結果を再利用しない。準備重複の1 red、別precision / profileとfixture resetを含め、監査・cache全6 passed / skip 0。これらのfakeは監査と準備回数の契約のみで、GPU性能や実モデル成功の代用にしない。

## 修正後の完了結果

[修正後の12条件・実backend・保存model hash](results/m3-vision-windows-revised-20261013.json)。同じEditorで2026-10-12 22:38:12〜22:50:18 UTCに全12条件が完了した。fixture SHA-256は `b50531a1844b238cb7ff3a95fd3099f6f2ff1e83a7b845c192927639cc6d1389`。CLI観測の`UNITY_EDITOR_UNRESPONSIVE` / `SafeToRetry=false`は保持し、再実行せず同一operationの保存報告から完了を確認した。

| 条件 | 結果と限界 |
| --- | --- |
| Float16 / CPU・GPU × 3 profile | 6条件合格。cosine最小 `0.9999999990336575`。GPUのactual worker / outputはGPUCompute、CPU計算 / 入力要求・中間readback flushは全GPU条件で0件。重み保存形式の精度であり、GPU前処理・最終embedding・他環境の成功ではない |
| FP32 / CPU・GPU × 3 profile | 6条件不合格。cosine最小 `0.9999999989783473`は条件を満たすが、要素単位許容差比は4.18〜8.99で超過。許容差を緩めず、段階・演算別の小さい再現redから原因を調査する |

今回の全体 `success=false`は維持する。初回のGPU6条件は数値推論前の監査失敗、修正後は実GPU数値を取得した結果であり、両者を区別する。初回約22分45秒と今回約12分5秒は、GPU数値推論の有無が異なるため同じ仕事量の高速化比較には使わない。import / 保存を各profile / precisionで再利用する契約は6件の小さいテストで別に確認している。

実行中にはEditor working set約27.6GBを観測したが、測定instrumentによる正確なpeak・解放後の収支は未取得。モデル・asset・worker・tensorの所有権と反復切替 / 解放をC4 / C7で検証する。FP32差、GPU前処理、共通本体への接続は引き続き必要で、PRをdraftのまま保持する。
