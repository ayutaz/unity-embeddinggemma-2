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

修正後の同じ12条件を実行中。初回失敗を上書きせず、修正後の実backendと数値・時間を別結果として回収する。FP32 elementwise失敗の原因調査、GPU前処理と共通本体への接続は引き続き必要である。
