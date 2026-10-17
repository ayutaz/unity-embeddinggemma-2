# 書き込み専用GPU出力のCPU割り当て・転送の削減

候補SDK: `d7b2233`。正方形画像の[GPU区間観測](encoder-gpu-dispatch-observations.md)で、BatchedMatmul / SoftmaxのCPU側経過時間が大きいことを確認し、出力の初期化経路を調べました。

## 問題と変更

Sentis 2.6.1の `new Tensor<float>(shape)` は `CPUTensorData` を既定でゼロ初期化します。GPU pin時にはこのCPU storageから変換し、`CPUTensorData.ConvertToComputeTensorData` が `ComputeBuffer.SetData` で転送します。既存のGPU演算は直後のshaderで出力の全要素を書き込むため、その初期データは不要です。

9つのGPU演算の出力を、storageを持たない `Tensor<float>(shape, (ITensorData)null)` から生成する共通helperへ変更しました。既存の `ComputeTensorData.Pin` が新しいGPU storageを割り当てた後、同じshaderをdispatchします。GPU演算の式・累積順序・dispatch範囲は変更していません。

対象はBatchedMatmul、BlockedMatmul、Gelu、Glu、ScalarDivide、SequentialScale、Silu、Softmax、Tanhです。重みや入力データのCPU→GPU転送と、演算結果の最終readbackは別の境界です。モデル・graphの再生成は不要です。

## 検証

[TDD・既存GPU回帰の監査](results/gpu-write-only-output-allocation-tdd-20261018.json)：

- 元のshape-only allocationで3件中2件が意図した失敗、1件成功、skip0、exit2。CPU storageの存在によるnull不一致を確認しました。
- 修正後は同じ3件すべて成功、skip0、exit0。形状・型・所有権と、CPU storageなしの初期状態を確認しました。この3件はモデルを使わないnographics試験です。
- D3D12 / RTX 4070 Ti SUPERの既存GPU演算・所有権・小さなgraph等の回帰は120 pass / fail0 / skip0 / exit0でした。実モデル全moduleの受け入れとは区別します。
- shader sourceのGit treeは修正前後とも `365b1cc6eded420c0c3246db2132170e383981ba`。既存の数値参照・許容差は変更していません。

## 固定した正方形画像の前後観測

[FP32の対応監査](results/gpu-write-only-output-fp32-square-correspondence-20261018.json)と[Float16の対応監査](results/gpu-write-only-output-float16-weights-square-correspondence-20261018.json)では、SDK以外の入力・descriptor・参照・精度・計測source `c2f766e`・Unity・GPU/API・camera/FPS・worker数・16GiB guardを照合しました。両形式の各4回で180 model区間・179 operation区間を全て観測し、計8組の768 float出力は修正前後でバイト一致しました。両実行は参照一致・解放・1 pass / skip0 / exit0です。

| 形式 | 修正前warm | 修正後warm | 修正後の続く2反復 |
| --- | ---: | ---: | ---: |
| FP32 | 19.689秒 | 5.214秒 | 4.832 / 5.481秒 |
| Float16重み | 34.048秒 | 5.067秒 | 5.009 / 4.603秒 |

これはそれぞれの固定入力の観測です。別のproject Library・起動時刻を使い、OS/shader cache・clock・background loadは制御していません。一般的な高速化率、平均性能、形式間の優劣、cold初回性能を示すものにはしません。profilerの影響とGPU queue時間の解釈も保持します。[FP32独立参照監査](results/gpu-output-dispatch-image-square-fp32-d7b2233-first-audit-20261018.json) / [Float16独立参照監査](results/gpu-output-dispatch-image-square-float16-weights-d7b2233-first-audit-20261018.json)。

## 残る受け入れ

正方形画像以外の固定入力でも新candidateの実GPU区間と参照vectorを確認します。環境・cache・profilerの影響と、kernel稼働時間／GPU queue時間／CPU側経過時間の違いを保持します。

続いて新しいSDK sourceで全6 moduleの元参照・採用Float16形式・混在検索の全順位・キャンセル／切替／解放・正常終了を確認します。最終candidateの導入・UI・公開取得・実マイクも別の残件です。現時点では高速化の達成、全実モデル互換、優先1〜3の完了は宣言しません。
