# M3: GPU画像前処理のTDDとWindows検証

> 設計・検証の履歴です。本文の状態・実行中・次の作業は記載当時のものです。現在の対応範囲は[検証状況](status.md)、残作業は[ロードマップ](roadmap.md)を参照してください。固定sourceの失敗・未実行も証拠として保持します。

2026-10-13。I3の製品前処理と重み不要の公式processor参照生成を追加した。画像encoder・共通本体・最終768次元API / sampleの合格とは分ける。正式Release / tagは作成しない。

## 入力と所有権

`ImagePreprocessor.Process(Texture)`はbatch 1の8bit RGB / RGBA Texture2D、作成済みsingle-sample 2D RenderTextureを借用する。sRGB保存値はSentisのexact-size TextureConverterでRGBへ戻し、linear保存値はそのまま使う。alphaは無視する。null、破棄済み、対応外format、未作成RenderTextureは推論前に拒否する。現入力契約は各辺1〜8192pxで、浮動小数・HDR・圧縮formatは明示拒否する。

公式profileはpatch 16px、pooling 3、最大280 soft token / 2520 patches。縦横比を保つ48px単位のresize、RGBの0〜1へのrescale、row-majorのpatchとpatch内pixel / RGB順、末尾zero padding、positionの末尾[-1,-1]を再現する。CPUで作るのは寸法・補間係数・位置のmetadataで、画像pixelは読み戻さない。

実装はGPU horizontal整数bicubic → GPU vertical整数bicubic / patch化。uint8 resizeの各pass後の丸め / clampは[固定PyTorchの係数・fallback実装](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cpu/UpSampleKernel.cpp)と[整数AVX実装](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cpu/UpSampleKernelAVXAntialias.h)を確認した。汎用bilinearへの置き換えは行っていない。

`ImagePreprocessingResult.Pixels`はGPUCompute float [1,2520,768]。独立した結果を所有し、次段GPU処理完了まで保持してDisposeする。processor再利用・Disposeは過去の出力を破棄しない。メインスレッドのみ。入力Textureはcallerが所有する。結果は作業用Tensor / coefficient bufferも保持するため、長期間の多件保持ではその容量を含めて測定が必要。

## Redと最初のGreen

- Python参照生成14 assertion red → 14 green。独立した公式processorとの全pixel / position、入力provenance、固定source / revision、hash、誤設定拒否、途中失敗での成功receipt除去を検証。最初のstubはKeyErrorになったため、success=falseのstubへ直して意図したassertion redを別実行した。
- Unityの入力・3 profileのGPU値 / padding・所有権はcompile成功、9 assertion red。公式参照の4経路は最初に測定用reflectionの型誤りで失敗し、その修正後に未実装による4 assertion redを別実行した。
- 最小実装後、同Editorで13 passed / failed 0 / skipped 0 / inconclusive 0、CLI exit 0。Unity 6000.3.21f1 / Sentis 2.6.1、NVIDIA RTX 4070 Ti SUPER / Direct3D12。
- 公式fixtureはsquare / landscape / portrait / 奇数寸法 / downsample / 同寸法の6ケース。Texture linear / sRGB、RenderTexture linear / sRGBの計24件で全pixel・position一致、最大pixel絶対誤差0。alphaを変えた入力、非対称のRGB pattern、非readable Textureも使用した。
- 出力をGPU Workerの全要素加算へ直接渡し、その最終出力取得後に元pixelを診断readbackした。次段の実GPU、計算 / 入力CPU fallback・readback flush 0。前処理はcustom compute dispatchで、Runtime内にpixel downloadはない。

この最初のTDD実行は既存consumerのEditor assemblyへsourceを配置した結果で、配布Runtime所属の証拠には数えない。source / log / 結果のhashは[receipt](results/m3-image-preprocessing-tdd-windows-20261013.json)に保存する。参照生成の固定sourceは `1fd94e28045c9bb4752bf92439f02912d57dd1f2`、torch 2.14.1 / torchvision 0.29.1 / transformers 5.19.0、元processor revisionは `914f7f89142e33e77833254d9c9b90c3cef7303b`。重み・encoderは読み込んでいない。

## 次の受け入れと未達

固定Git `c21f1f8709b196c55c2b707a389dccc37437b21f`へ既存consumerを更新し、source / shader overlayを退避した。両公開型はEmbeddingGemma.Runtime、112 canonical Git blobs + package.json semantic一致、113ファイル不一致0。package 117 passed / skip 0、公式4 test / 全24ケースpassed、最大pixel差0。更新要求は受理後EOF / SafeToRetry=falseで切断したが、再要求・Editor再起動を行わず同PIDのReady、requested / resolved SHA一致を確認した。[固定Git受け入れreceipt](results/m3-image-preprocessing-git-windows-20261013.json)。最終candidateの新規空consumerや画像encoder接続とは分ける。

この後、同固定Git Runtimeへ8192×8192のGPU生成単色RenderTextureを渡す追加テストが「pixel 0: expected 0.2, got 0」で失敗した。Editor logは「maximum thread group count 65535」の超過を記録。横passは73728 groupsを要求していた。dispatchを2次元へ分割し、shaderのlinear indexもrow offsetを含む形へ修正した。係数・補間・数値許容差は変更していない。

最初の修正後は全pixel比較に到達したが、red観測用ignore設定が生成したLogにNoUnexpectedReceivedが失敗。ignore設定を除去すると既存のshader signed integer division Warningが検出された。非負の座標 / strideをuintにし、係数とsumはsignedのまま保持した。最終5 passed / fail 0 / skip 0 / CLI exit 0、最大寸法の全patch / paddingと通常の公式24ケースが合格した。これは修正source / shaderのoverlay実測で、上記c21のGit結果へ付け替えない。[境界修正TDD](results/m3-image-preprocessing-max-input-tdd-20261013.json)。警告の解消だけで速度改善とは判定しない。

修正後の固定Git `71284e302d5e3e643e4ed98272c5774640582c1e`でsource / shader overlayを退避し、requested / resolved SHA一致、114 canonical blobs + package.json semantic一致の115ファイル不一致0を確認。公開型はRuntime assembly、shaderもpackage内assetへ解決した。最大入力を含むpackage 118 passed / fail 0 / skip 0 / CLI exit 0。公式4 test / 24ケースは全passed、最大pixel差0、position一致、次段GPU Workerの計算 / 入力fallback・readback flush 0。Resolveは受理後EOFで再要求せず、同EditorのReadyを追跡。readonly型probeの引用符誤りによるCS0103は補助検証の失敗として保存し、訂正後の成功と分ける。[修正後Git receipt](results/m3-image-preprocessing-final-git-windows-20261013.json)。

次にI2の画像FP32不合格を解消し、実encoderへのGPU接続、公式prompt、最終API / sample、Playerと必須他OSへ進める。公式画像前処理fixtureの生成・artifact保存は既存Actionsへ接続した。CIの成功はUnity実GPU・実encoder成功と分ける。

今回の結果は上記6入力寸法と最大寸法の単色画像、Windows backendのみ。全寸法のメモリ上限・極端な縦横比、長い反復での資源 / 性能、Player strippingと他OSは未検証。直接のGPU profiler captureも未取得で、転送の判断はsource経路、GPU Tensorと次段Workerのfallback / flush監査の範囲。これらを成功へ読み替えない。
