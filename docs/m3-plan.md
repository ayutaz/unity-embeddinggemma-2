# M3: 画像対応のTDD計画

更新日: 2026-10-13。**画像end-to-end API / sampleはmainに未実装、実画像encoder / Unityの合格証拠は未取得**。共通backboneの一部はPR #34で実装済みだが、合成特徴量によるPython照合は画像encoderの受け入れを満たさない。依存・実行場所・公開条件は[全体計画](multimodal-tdd-plan.md)、既存仕様調査は[技術調査](technical-approach.md)。M3完了時には正式リリースせず、M4と全必須環境の検証へ進む。

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
