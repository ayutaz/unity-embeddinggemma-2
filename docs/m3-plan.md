# M3: 画像対応のTDD計画

> 文書整合確認: 2026-10-15。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。冒頭の現在方針が、下の旧checkpointの着手順より優先する。

## 2026-10-15の画像の現在地

GpuGeluの所有権 / GPU専用 / 再利用を5 Redから実装。捕捉gate7,741,440値に対する現CPU式のbit差は既存3,048,817→候補6,716、許容差超過0で全bit一致ではない。元layer 0への接続は関連16 Green・元native許容差超過0 / source bytes保持。全体の112 RMS＋16 GELU＋patch384候補square-0は571要素超過、host peak約9.252GiB・VRAM snapshot15,638MiBで未採用。[固定receipt](results/m3-gelu-gpu-layer-and-full-failure-audit-fix-20261015.json)。全体反復を止め、次はstageごとの未使用GPU cache解放をopt-inで局所TDDし、attention / softmaxを独立照合する。係数監査の46.802秒は、同じ全係数bytes / digestを保持するWindows CNG経路へ置換し、3 Red→関連10 Green、単体0.467秒を確認した。全推論のbenchmarkや画像合格へは流用しない。初回実GELU fixtureのreshape前提失敗と監査testのcompile失敗は別記録。既存GPU演算 / 音声経路は不変。残る元RMS、全3 profile、最終API / sample / Playerは未完了。以下は前回checkpoint。

RMSを全体へ戻す際のWorker増加を抑えるため、除算をFP32残差で補正するopt-in exportを追加。3 Red→関連79 Green、現CPU式の全1,935,360値へ1 Workerの実GPU全bit一致。元layer 0への7 RMS組込みも、1 Red→関連20 Greenで前回のqualified GPU出力bitを全保持し、元native layer gateの超過0、segmentは15→8。[固定receipt](results/m3-inline-rms-layer-equivalence-and-whole-failed-20261015.json)。全16層の112 RMSを組み込んだsquare-0は577要素超過・最大比7.556308で不合格。元RMS1か所は未変更、他5caseは未実行。host peak約4.331GiB、GPU実行＋取得4.142秒だが、準備46.964秒には係数hash監査等も含むので製品の純粋な準備benchmarkではない。この全体候補は未採用・反復停止。次は活性化とattention積和の局所参照を確認してから元全profileへ戻す。最終API / role / mask / 寿命 / text回帰 / sample / Playerは未完了。以下は前回checkpoint。

元layer 0の全7 RMSと7投影をGPU接続し、固定元layer参照の許容差超過0・最大比0.766697へ到達。bit差1,518,970は残り、全bit一致ではない。4次元境界 / 元scale借用の5 Red→関連22 Green、全RMS marker / callable headの4 Red→27 Green。768幅4か所と64幅3か所は捕捉入力上の現CPU式へ実GPU全bit一致、元model / scale bytes・参照 / 許容差を維持した。[固定receipt](results/m3-seven-rms-original-layer0-gpu-20261015.json)。追加fixtureのcompile失敗 / XML欠測は別証拠として保持。既存Runtimeは未変更。画像全体は再実行せず、以前のsquare-0の566要素超過が未達のまま。次はGELU / attention積和を局所照合する。小さいGELU CPU probeは分離式にも55 bit差が残り未採用で、GPU成功や元内部参照の再現とは扱わない。全3 profile / 実API / sample / Playerの範囲は維持する。以下は前回checkpoint。

元layer 0の7 RMSを、通常GPU最終bitと旧hashを保持して捕捉（1 pass / skip0）。64幅3か所は新しいscaled / unscaled exportで13 Red→関連30 Green・音声算術 / package込み76 Green、捕捉元入力と元scaleを使う現CPU式へ実GPU全bit一致（3 pass / fail0 / skip0 / fallback0 / flush0）。768幅4か所も既存exportのCPU式全bit一致だが、GPU確認は最初の1か所のみ。[固定receipt](results/m3-all-rms-observation-and-head-gpu-20261015.json)。次は4次元unary境界 / 元scale借用をTDDで実装し、残る768幅3か所をGPU照合してから7 RMSを元layer 0へ戻す。元native内部参照や全encoder合格とは扱わず、下記wholeの566要素不合格と全profile / 最終API / sample / Playerを未達のまま保持する。以下は前回checkpoint。

最初のRMS＋7つのblock384投影を元layer 0へ戻した実GPU gateは、元参照・係数・入力・許容差のまま超過0。境界 / RMS単体を含む15 pass / fail0 / skip0。全bit一致ではなく、画像全体合格へは流用しない。確認済み最初の層だけを元全encoderへ接続したsquare-0は566要素超過・最大比6.791765で1 fail / skip0。10 segment、host peak約4.017GiB、実行＋取得3.660秒だが、定常性能合格ではない。[固定receipt](results/m3-first-layer-rms-reconnection-20261015.json)。次は全体反復を止め、7か所のRMS入力 / 出力 / scaleを通常GPU最終bitを保持して捕捉し、残る64 / 768幅の正規化・活性化・積和を局所照合する。元3 profileの全条件、実Texture→768 API、role / mask / 寿命 / text回帰 / sample / Playerは未完了。以下は前回checkpoint。

最初のRMS単体は、元patch / scaleと現CPU式を使い、実GPUの全1,935,360値で全bit一致した。原因はSentisが除算768→epsilon加算をScalarMadへまとめた丸め差で、明示GPU除算の境界を保持して修正。元RMS観測の再現・失敗候補・統計FMA診断・1 pass / fail0 / skip0を[固定receipt](results/m3-first-rms-gpu-explicit-division-20261015.json)へ分離した。元native内部参照は未取得で、画像encoder合格には数えない。次はこのRMSを元layer 0へ戻し、元固定参照で改善を検証する。他のRMS/活性化、全3 profileの全条件、最終API / role / mask / 寿命 / text回帰 / sample / Playerは未完了。以下は前回checkpoint。

現在の次工程はRMS正規化の固定GPU照合。元patch入力のlayer 0は通常4要素超過、7投影のblock384候補は1要素超過。全114投影を元square-0 encoderへ戻しても579要素超過・最大比5.757522で未合格だった。実行＋取得30.978秒・host peak10.969GiBのため、この全体候補を反復せず小さい正規化/活性化テストへ戻る。768幅平均は7 Red→CPU/ONNX関連25 Green、元patch64行の平均もbit差0。有界1万点のCPU probeでは補正inverseがnative powerへbit差0だが、GPU成功・元全context合格へは数えない。[固定証拠と残件](results/m3-layer0-static-gpu-and-rms-next-20261015.json)。以下は前回checkpoint。

元画像のpatch prefixで1,256,626 bit差を再現し、係数を保つGPU block384投影で元1,935,360値へ全bit一致した。128/192/256/512/768の不一致も保持する。sliceの入力番号問題を小さいCPU/GPU Redから修正し、GPU投影・slice関連24 Green、package22 Green。[固定receipt](results/m3-patch384-gpu-and-original-encoder-20261015.json)。ただし、この投影を元全encoderへ戻したsquare-0は579要素超過・最大許容差比7.406214で不合格。局所一致を全体合格へ置き換えない。次は正確なpatch出力を起点にlayer 0の残る投影・演算を切り分け、元3 profileの全条件を確認してから実Texture→正規化768 APIへ進む。画像全chain / role / mask / 所有権 / text回帰 / sample / Playerは未完了。以下は前回checkpoint。

GPU画像前処理はmain、ImageFeatureEncoder / ImageEmbedderはDraft実装・合成契約合格。画像PythonのQuery / Document / Rawは3 profile・6画像×3 roleで18件合格した。[receipt](results/m3-real-image-all-role-reference-20261014.json)。元FP32 encoderの固定gateとUnity実Texture→768次元の全chainは未達。

補償投影候補の元12数値caseは全件不合格。最後のnorm4実行はユーザー中断で、square / landscape CPU / GPUは不合格、portrait CPUは途中、portrait GPUは未開始。[中断receipt](results/m3-vision-norm4-user-interrupted-20261014.json)。小さいGPU契約やPython role参照でこの未達を置き換えない。

再開は[全体計画R0〜R5](multimodal-tdd-plan.md)に従う。対象fixtureのGPU専用変更をcompileし、一つのGPU入力で参照文脈・数値差と準備負荷を確認する。元3 profile、FP32 / 採用Float16、全role・実Texture / RenderTexture・所有権・text回帰・sample・Player / 必須環境を保持する。層数25265への展開を性能合格の候補と扱わない。

## 以前の画像計画・固定sourceの履歴

更新日: 2026-10-13。**GPU画像前処理はPR #52でmainへ統合・Windows単独受け入れ済み。画像end-to-end API / sampleと実画像から最終768次元の受け入れは未完了**。画像encoder / projectorのPython実モデル照合はPR #37で合格した。Sentis encoderはPR #41のWindows実測でFloat16のCPU / GPU全6条件が合格、FP32全6条件は固定elementwise許容差超過で不合格。共通RuntimeのPR #43はmainへ統合済み。下記の実測sourceは過去の固定headとして維持し、統合headへ付け替えない。[全体計画](multimodal-tdd-plan.md)の工程3として進め、M3完了時には正式リリースせずM4と全必須環境の検証へ渡す。

I1 / I2の固定profile encoder / projector exportと公式processor参照・Actions gateをTDDで実装した。[実装と検証範囲](m3-vision-export-validation.md)。小さい回帰66 passed / skip 0。[実モデルPython CI 38310718250](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38310718250)の3 profile / 6ケースでminimum cosine `0.9999997615814209`。出力は512次元soft tokenであり、画像の最終embeddingではない。

### 履歴checkpoint

公式prompt / mask契約PR #55と最終画像参照PR #56はmainへ統合済み。実モデルPythonで3 profile / 6画像の768次元、native feature → 保存backbone、保存encoder → 保存backboneを固定許容差で照合した。producer `35d357263ae29de4db1ad93d1cac3616ea1e61b4`、[CI 38356261690](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38356261690)。最新baseのhead `9814ce9d8bd70fa07cbd191867d93d7f4dd461f5`の[CI 38364660005](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38364660005)全8成功を確認して統合。Python参照の合格と、画像Runtime・Sentis全chainの未実行は別に記録する。

次はzero-audio profileのimport、入力 / feature順・mask / 所有権・途中失敗のred → greenからGPU前処理・実encoder・共通本体を接続する。元のFP32全6条件の不合格を解消し、最終768次元・全順位・画像検索sample・Playerへ進む。下記の過去の実測sourceは維持し、C3の公式prompt構築の未完了記述はこの最新checkpointで更新する。

### 直近のTDD順序

I3の画像GPU前処理は[PR #52](https://github.com/ayutaz/unity-embeddinggemma-2/pull/52)で実装。固定Git Runtime `71284e302d5e3e643e4ed98272c5774640582c1e`で全package 118 green、Texture / RenderTexture × linear / sRGBの公式24ケースは最大pixel差0。8192pxのdispatch不具合も1 redから修正し、最大入力・paddingを確認した。[実測と未達](m3-image-preprocessing-validation.md)。次はI2の数値差修正と、統合済みC3公式prompt / maskへの実encoder接続。I4の実モデル連結の合格判定はI2 / I3 / C3が揃ってから行う。GPU profiler、長い反復 / memory、Playerと他OSは別gate。最終API・sample・PlayerもTDDで実装し、encoder単独の合格でM3を完了にしない。

1. I3のGPU前処理実装と単独Windows照合は完了。次の実encoder接続では、Tensor寿命・途中失敗・中間readbackを新しい契約のredから確認し、GPU profiler / 長い反復memoryの証拠を追加する。
2. I2のFP32差をstage / 演算別に特定し、独立参照との不一致を小さいテストでredにしてから修正する。現在の `rtol=1e-4` / `atol=1e-5`は緩めず、3 profile / 全12条件の実モデル再照合と保存 / 再読込を確認する。
3. C3の[PR #43](https://github.com/ayutaz/unity-embeddinggemma-2/pull/43)、確認head `21e598c5b53d5ad73557ee6d606b12770254d73b`では配置・cache修正の68件greenに加え、実1024長backboneのCPU / GPUCompute × FP32 / Float16全4条件が合格した。[固定headの結果](https://github.com/ayutaz/unity-embeddinggemma-2/blob/21e598c5b53d5ad73557ee6d606b12770254d73b/docs/results/multimodal-backbone-sentis-windows-20261013.json)。最初の4条件はconsumer Editor assembly内source、後続の固定Git Runtime更新consumerも4条件合格した。[Git更新のreceipt](results/multimodal-git-runtime-upgrade-windows-20261013.json)。いずれも合成project済みfeatureの受け入れであり、実画像feature接続・統合済み公式prompt / maskとの連結・最終candidateの空consumerは別gateとする。placeholder / feature数・順序の不一致をred → greenにし、実画像の最終vectorを照合する。
4. I4〜I6のAPI → 横断検索sample → Playerを下表の契約順でTDD実装する。最終768次元と全順位、実操作・解放・正常終了の必要条件を満たすまでM3は未完了とする。

PR #53の実Sentis診断は2026-10-13 04:56 UTCにCPU / GPUの2テストが独立参照の文脈確認で不合格。case_0のtrace最終値は元保存Sentisと一致したが、新native参照と元の固定参照の一致gateが未達で、case_1は未実行。原因を確定したとは扱わない。元 / 新参照の入力・環境を照合し、途中の成功flagを拒否する報告監査のredを追加する。診断文脈の合格後に演算別red → 最小修正 → 全12条件へ進む。[今回のcheckpoint](results/tdd-acceptance-plan-checkpoint-20261013.json)。

[12条件の実測](results/m3-vision-windows-plan-snapshot-20261013.json)はencoderのみ。全GPU条件で実出力はGPUCompute、CPU計算 / 入力要求・readback flushは0件だが、GPU画像前処理と共通本体へのfeature受け渡しは別に検証する。CLI観測timeoutと実operationの保存報告を区別し、未変更の重い検証を再起動しない。

## 範囲と参照固定

`Texture` / `RenderTexture` → GPU前処理 → 画像encoderのsoft token → 共通本体 → 768次元を実装する。encoder出力を直接L2正規化して最終embeddingとする実装は採用しない。入力は最初に単一画像 / batch 1と参照で確定した解像度profileを扱い、縦横比・許容寸法・上限・複数画像の対応範囲をAPIで明示する。

最初に固定revisionの公式processorからresize計算・補間・色空間 / RGB・alpha扱い・値域・patch順・位置ID・pooling・padding / mask・soft token数・promptを確認する。既存調査のpatch 16px / 標準280 tokenは調査値であり、Unity profileの採用値はこの確認後に固定する。共通本体C3の長さ拡張なしに128長へ切り詰めない。

参照fixtureは小さい単色・勾配・checkerboard・縦横非対称画像、縦長 / 横長、奇数寸法、RGBA・RenderTextureを用意する。これらは前処理の検出用であり、意味検索の評価は別の権利確認済み固定画像 / 日英queryを使う。参照生成はActions / tools / uv、重みと大きい中間テンソルはartifactへ保存する。

## 実装順とRed / Green

| ID / 依存 | 先に観測するRed | 最小実装 | Greenと実モデルの受け入れ |
| --- | --- | --- | --- |
| I1 参照・C2 | processor / profile / revisionの違い、patch位置・mask・期待feature数不整合をfixture監査が見逃す | processor参照CLI、fixture manifestと段階別参照・encoder export | 小さいmanifest検査green、Actions実モデルの前処理・encoder・最終vector参照とhashを固定。数値許容差を根拠付きで先に決める |
| I2 export・I1 | 実重みを使うexportの出力数 / 並び / paddingが固定Python参照と不一致。互換性の小さいgraphテストも先に作る | encoderを対応演算へ展開し`.pt2` export、import / `.sentis`保存・採用Float16重み形式 | Actionsで元Python対export照合、Unity CPU / GPUで実encoder feature照合。import成功だけで合格にしない |
| I3 GPU前処理・I1 | 向き反転、sRGB / linear混同、resize丸め、channel / patch順、padding値で参照と不一致。借用Texture破棄の契約もred | GPU上のTexture変換・resize・patch / position処理。固定profileで始める | 合成画像の中間tensor比較と実GPU実行、前処理のCPU readbackなし。Texture / RenderTexture両経路で確認 |
| I4 共通本体・I2 / I3 / C3 | image placeholderとfeature数不一致、feature順違い、prompt欠落、mask / projection / pooling違いで最終vector不一致 | featureをGPU tensorのまま共通本体へ渡す。仮API名 `ImageEmbedder`、名称・所有権は契約PRで確定 | 実モデルfp32 cosine >= 0.999 / 採用Float16重み >= 0.99、有限768次元・単位長。CPU / actual GPUを別記。既存text128回帰 |
| I5 検索・sample・I4 / C5 | text→imageの全順位 / score、同点ID順が不一致。未準備、対応外画像、欠落モデル、停止後再操作の契約がred | 画像事前embeddingとtext query、結果画像 / score表示、入力拒否・解放 / 復帰 | 固定日英queryの全順位一致、実UIで準備→検索→表示→解放。画像向き・表示・エラー後再操作も確認 |
| I6 Player・I5 / C4 / C6 | encoder欠落 / hash破損 / profile違い・GPU fallback・途中結果をpassにできる。繰返し切替で寿命契約違反がred | bundle / cache監査、Player harness、画像sampleの配置・測定 / 終了監査を接続 | Windows IL2CPP / Release / High strippingで実モデルCPU / GPU、独立2起動の実終了0、cache・切替 / memory。他OSは全体公開条件へ |

I2のSentis非互換は演算子とshapeを保存して調査する。標準演算への分解を先に小さいPython / Unity比較で検証する。標準ONNXが有利なら別PRで同じ参照へ照合し、native runtimeプラグインを追加しない。

入力のnull、破棄済み / 対応外Texture、範囲外寸法、feature数・profile不一致は推論前に明示的に拒否する。自動crop / 切捨て / CPU fallbackで期待値を合わせない。画像upload・最終vector取得と、中間GPU tensorを区別してprofilerで確認する。

## 測定・完了条件

- I1〜I6の意図したred / green、source / model / processor / graph hash、Unity / Sentis、requested / actual backendを記録する。
- 前処理、encoder、共通本体、load / 初回 / warm反復、切替 / Disposeの時間・メモリと測定APIを残す。画像入力のCPU readbackと中間転送がないことを実GPUで確認する。
- 実画像の参照embeddingとtext→image全順位、既存テキスト検索が合格し、空Git consumerのsampleを実操作できる。小さい画像テストのgreenで実encoder互換性を完了にしない。
- Float16重みは実測した採用形式だけを宣言する。OOM / GPU非対応 / 他OS未実行を別記し、全体の正式公開条件へ引き継ぐ。

PRはI1 / I2の参照・export、I3のGPU前処理、I4 / I5のAPI・sample、I6のPlayer接続に分ける。各変更のgreen後に影響する実モデル照合を行い、変わっていない大きいモデル生成・Unity起動を繰り返さない。
