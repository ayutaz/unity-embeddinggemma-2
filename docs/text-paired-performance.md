# Windows Playerの既存テキストCPU / GPU比較

実行source `3a02fc0` のWindows IL2CPP / Release / High Player 2起動には、
両重み形式のCPU / GPU比較が保存されています。モデル実行を繰り返さず、
元report・正常終了記録・参照・全条件・保存した時間を再照合しました。
現在のSDK source `7d22168` とcompiled SDK入力の一致も確認しています。
[要約と元evidence hash](results/text-player-paired-performance-readback-20261020.json)に
8条件と各3回のwarm測定を保持しています。

| 起動 | 重み | CPU warm 3回の範囲 ms | GPU warm 3回の範囲 ms |
| --- | --- | ---: | ---: |
| first | FP32 | 553.677–607.449 | 29.778–36.825 |
| repeat | FP32 | 451.455–458.950 | 29.917–31.281 |
| first | Float16重み | 839.623–1136.350 | 44.038–53.051 |
| repeat | Float16重み | 802.269–1103.513 | 35.696–36.610 |

各起動内で同じ固定query・モデルをCPU / GPUへ渡した、同期API全体のwall時間です。
tokenize、入力準備、GPU待ち、最終readbackを含み、kernel専有時間や一般的な速度比では
ありません。CPU条件をGPUの合格に数えません。FP32 / Float16重みの差を全演算FP16化や
高速化の保証にもしません。

最初の1回もCPU / GPU間では同じcaseですが、warm queryとは異なる入力です。
その初回値はJSONへ別に保持し、上表のwarm範囲へ混ぜません。2回目のPlayerも新しい
process / Workerで、OS cacheを空にした実行ではありません。CPU→GPUという条件順序と、
初期化・cacheの影響を取り除いた純粋な比較とは区別します。

各条件で全15 text case、6文書・4検索、実backend、元cosine・順位・score、Worker解放を
元protocolが確認しました。元のraw vectorを新しく採取した監査ではなく、保存した
score等の独立照合とruntime内vector検査を分けています。破損reportの対照は両起動で
計34件を再度拒否しました。

元のWindows機能検証・Player sourceを保持し、新candidateでの再実行、画像・音声CPUの
受け入れ、SDK専有allocation、最低hardware、性能全体や正式公開の合格には広げません。
残件は[Windows先行計画](next-release-plan.md)で追跡します。
