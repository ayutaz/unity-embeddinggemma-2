# 最終画像embedding APIの合成受け入れ

2026-10-14 JST。`IImagePreprocessor` / `IImageFeatureEncoder`と、画像専用`ImageEmbedder` / `IImageEmbedder`を追加した。音声API PR #62の`IMultimodalBackbone`を共通契約として利用する。画像encoder単独のPR #64を含む後続branchであり、main未統合。正式Releaseは作成しない。

Textureと全stageは借用し、各callの前処理結果・projected soft tokens・empty audio tensorはAPIが解放する。固定profileのimage feature数、audio 0、prompt profile、全stage backendの一致をconstructorで検査する。`BuildImage`はtokenizer設定の画像markerから公式promptを構築し、role / titleの検査をGPU処理より前に行う。encoderとbackboneへGPU Tensorを渡し、最終768次元vectorだけをCPUへ返す。出力の長さ・有限性・L2 normを検査し、返却配列は独立したcopyとする。API破棄は借用stageを破棄しない。

| 検証 | 固定source・結果 |
| --- | --- |
| red | requested / resolved Git `2bce67fa1474ba6439ee912693e07309f639af65`。compile error / warning 0後、24 assertion failed / passed 0 / skip 0 / CLI exit 1。2026-10-13 15:45:27 UTC完了 |
| green | requested / resolved Git `c2c5516e83339450bf7ff4c65432ef9ddae06861`、Runtime overlayなし。24 passed / failed 0 / skip 0 / CLI exit 0。15:48:15 UTC完了 |
| package回帰 | 同sourceで225 passed / failed 0 / skip 0 / CLI exit 0。15:49:22 UTC完了 |
| compile | error 0 / warning 0 / CLI exit 0 |

redは最終API / image-only promptの不足、実ImagePreprocessor / ImageFeatureEncoderがstage interfaceを満たさないassertionである。小さい独立encoder / backbone graphでCPU / 実GPUの最終vectorを数式へ比較し、借用Texture保持、結果の独立copy、前処理 / encoder / backboneの途中失敗と再操作、各中間結果の破棄、null・異backend・profile不一致・不正role / title・不正最終出力を検証した。Editorは6000.3.21f1、Sentis 2.6.1、RTX 4070 Ti SUPER / Direct3D12、既存consumer `artifacts/k`、PID 149180の1つ。公式Test Framework 1.6.0のembedded条件を保持する。[機械receipt](results/m3-image-embedder-composition-windows-20261014.json)。

CPUの前処理stageは独立fixtureであり、実TextureからのCPU前処理を実装・検証したことにはしない。実ImagePreprocessorは現在GPUCompute専用。この24件は合成graphの受け入れであり、実vision model / 最終backboneを通したTexture全chainは未実行。元のvision FP32 elementwise不合格、実画像の全profile / role、sample / Player / 必須環境・性能 / memory・最終空consumerは残っている。数値許容差は変更していない。

監査済み画像専用modelを用意した後のGPU使用例:

```csharp
using var preprocessor = new ImagePreprocessor();
using var encoder = new ImageFeatureEncoder(encoderModel, BackendType.GPUCompute,
    new ImageEncoderProfile(768, 768));
using var backbone = new MultimodalBackbone(backboneModel, backboneProfile, BackendType.GPUCompute);
var prompts = new MultimodalPromptBuilder(tokenizerJson, tokenizerConfigJson, backboneProfile);
using var embedder = new ImageEmbedder(preprocessor, encoder, backbone, prompts);
float[] vector = embedder.Embed(borrowedTexture, TextRole.Document);
```

`encoderModel`には同じ寸法 / 座標が焼き込まれていること、`backboneProfile`のimage feature数が256・audio feature数が0であることが必要。入力画像の前処理後寸法がprofileと異なる場合は拒否する。この例は動作検証済みsampleとしては扱わない。
