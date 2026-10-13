# CPU画像前処理と実測・未達条件

2026-10-14 JST。`ImagePreprocessor(BackendType.CPU)`を追加した。既存のparameterless constructorはGPUComputeのまま。CPUでも公式の寸法・bicubic antialias・uint8の横pass後clamp・patch順序・位置情報・zero padding・RGB rescaleを使用する。横passの中間値はbyte配列で保持する。CPU resultは独立したCPU Tensorを所有し、Textureは借用する。

読み取り可能なTexture2Dは`GetPixels32`から取得し、Unityのbottom-up格納を公式top-downへ並べる。GPU-only Texture / RenderTextureは入力を1回読み戻す。読み戻し非対応のRGB24 / BGRA32は、同じtransfer encodingの一時RGBA textureへ`Graphics.ConvertTexture`後に読み戻す。一時Textureは所有して破棄し、借用Textureやactive RenderTextureは変更しない。入力変換・読み戻しにGPUを使う条件と、CPU上のresize / patch化 / 後段Tensorを区別する。GPUCompute経路には途中CPU readbackを加えていない。

## 固定sourceの検証

| 条件 | 結果 |
| --- | --- |
| TDD red | test `e6bb946b4bd9cca3bcb8c55405f81e91f2867f59`をtest-only consumer assemblyへ配置。Runtime Git `c2c5516…`は変更せず、compile成功後にCPU constructor不足の12 assertion red / skip 0 / CLI exit 1 |
| 最初のgreen | Git `b5dba83cb9d5c3f73ff1fd6abfa4ebe26e11405f`で12 pass / failed 0 / skip 0 / CLI exit 0 |
| 追加形式のred・修正 | RGB24 / BGRA32のGPU-only inputが拒否され、最初の追加試験は2 pass / 3 fail。destination formatだけの指定でもsource formatエラーが残った。sourceをRGBAへ変換するGit `10fa49cd4a7db231cb3e27434dfabaf47b54fb1f`でCPU12件＋形式4件は合格。追加chain数値1件は不合格を保持し、計16 pass / 1 fail / skip 0 / CLI exit 1 |
| 公式CPU比較 | 同Git、harness `71952846f90b30fa8f3b7cb4b6cef4ba6efd10df`。公式6画像 × 10入力経路の60条件が全合格。pixel最大差0、position完全一致、CPU consumerへ渡した後の値も固定atol 1e-6内。NUnit10 pass / fail 0 / skip 0 / CLI exit 0、2026-10-13 16:23:57 UTC完了 |
| GPU回帰 | Git `b5dba83…`、既存harnessの公式6画像 × 4入力経路24条件が全合格・最大差0。NUnit4 pass / fail 0 / skip 0 / CLI exit 0。後の修正はCpuColorsのみで、この結果を新sourceでの再実行に付け替えない |
| package回帰 | Git `10fa49c…`で237 pass / failed 0 / skip 0 / CLI exit 0、16:26:45 UTC完了。Assetsの追加acceptance失敗をこのpackage-only成功で消さない |
| compile | Git `10fa49c…`でerror 0 / warning 0 / CLI exit 0 |

公式入力は既存Actions artifactのproducer `1fd94e28045c9bb4752bf92439f02912d57dd1f2`、model revision `914f7f89142e33e77833254d9c9b90c3cef7303b`、reference SHA `8076c38ded6c2486082092c0f0a43ec553760280ac46d8b98bfaccce829c3334`を再監査して使用した。新しいモデルdownload / exportは行っていない。10経路はreadable RGBA、GPU-only RGBA、RenderTexture、GPU-only RGB24、GPU-only BGRA32のlinear / sRGB。fixtureはsquare / landscape / portrait / odd / downsample / identity。大きい入力・出力はGitへ含めない。[全60条件](results/m3-image-cpu-official-windows-20261014.json)。

Editorは6000.3.21f1、Sentis 2.6.1、RTX 4070 Ti SUPER / Direct3D12、既存consumer `artifacts/k`でrequested / resolved Git一致、Runtime source overlayなし、Editor本体PID 149180の1つ。公式Test Framework 1.6.0のembedded条件を保持する。fresh consumer、Player、他OS / 実機、GPU profiler、性能倍率・長時間memoryは未検証。

## 合成最終chainの数値差は未達

実TextureのCPU前処理から独立した小さいencoder / backbone graphを通す追加試験は、最終値 `0.457341015` が独立した高精度計算 `0.45733540469874051` から固定atol 1e-6を超えて差があり不合格。[初回の形式・数値失敗](results/m3-cpu-image-integration-initial-failed-20261014.json)、[destination指定だけでは解消しなかった失敗](results/m3-cpu-image-destination-format-failed-20261014.json)、[形式修正後16 pass / 1 fail](results/m3-cpu-image-conversion-and-integration-20261014.json)を保持する。

harness `2018623841a7b061fa205c3101299304d02d855e`による[段階診断](results/m3-cpu-small-chain-stage-diagnostic-20261014.json)では、前処理のfloat32格納値から算出した期待meanと実pixelのdouble accumulation meanは両方 `0.22946780068533762`。encoder出力meanは `0.2294643372297287` で差 `-3.4634556089208068e-6`。最終vectorから推定したseedと観測encoder meanから算出するseedの差は `1.1459288905335541e-5`。前処理で初めて差が生じた証拠ではなく、これらのgraphのreduction / 後段計算の局所化が次の課題。これは実vision modelではない。元のテスト・許容差は変更せず、graphを置き換えて元の失敗を消していない。

実画像encoderのFP32不合格、実model全chainの最終768次元照合、role / sample / 必須環境は残る。CPU前処理60条件の合格だけでM3や工程6の完了にはしない。

入力読み取りの仕様はUnityの[GetPixels32](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Texture2D.GetPixels32.html)、[AsyncGPUReadback.Request](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Rendering.AsyncGPUReadback.Request.html)、[WaitForCompletion](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Rendering.AsyncGPUReadbackRequest.WaitForCompletion.html)、[ConvertTexture](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Graphics.ConvertTexture.html)を確認した。同期readbackはstallを伴うため、GPU-only入力での時間 / memory測定とsample操作の受け入れを残す。

## 最新Required CIのPython vision不合格

head `da1e68d`の[CI 38431029093](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38431029093)はPythonの`Validate official image processor and real vision encoder / projector exports`で`ValueError: vision export mismatch: square-0 cosine=0.9999997615814209`、exit 1となり、Required CIも不合格。Python4環境 / package / actionlintは成功した。このPRの差分にはPython / workflow変更はなく、CPU前処理のC#試験とは別の未達として保持する。cosineの表示値だけでelementwiseを合格にしない。原因は未確定であり、許容差を変更せず、失敗runを一時障害として再実行して消すことはしない。
