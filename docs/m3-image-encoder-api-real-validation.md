# 実vision modelでの画像encoder API受け入れ

2026-10-14 JST。TDDで実装済みの公開`ImageFeatureEncoder`を、既存の監査済み実modelへ接続した。Runtime Gitは `10fa49cd4a7db231cb3e27434dfabaf47b54fb1f`へ固定し、requested / resolved一致、Runtime overlayなし。モデルの再取得・再import・保存は行わず、6つの保存済み`.sentis`をsize / SHAで照合して使用した。

## 全12条件の結果

Python producerは `e2a6dc01fbca8c9be1f3bbe875325ad338de0ca0`、reference JSON SHAは `3dfccd1a487ce04df3f8d640483ca045f7a7a56231bc1096ad0f845c923a29c6`、validation JSON SHAは `ab1aa775b553f33a5e2892abed2cdcef7768e35fbad6fa5af9ec3286a7ba90ba`。元の保存model receiptとして、以前の全12条件report SHA `720e4b27049d04005108c75525ce7aa2c895b33cdcfc2021915d3c3440e9ae91`を固定した。referenceの全file hashも再照合した。

| 保存weight / backend | profile数 / 画像数 | 結果 |
| --- | --- | --- |
| FP32 CPU | 3 / 6 | cosine合格、固定elementwise不合格。最大許容差比8.993760372330813 |
| FP32 GPUCompute | 3 / 6 | cosine合格、固定elementwise不合格。最大許容差比8.352316839011602 |
| Float16 CPU | 3 / 6 | 元のFloat16 cosine gateに全合格。最小cosine 0.9999999991103836 |
| Float16 GPUCompute | 3 / 6 | 元のFloat16 cosine gateに全合格。最小cosine 0.9999999990336575 |

FP32は元のcosine >= 0.999かつ `atol=1e-5` / `rtol=1e-4`、Float16保存weightは元のcosine >= 0.99を維持した。Float16のFP32 elementwise許容差比は診断値として保持し、elementwise合格とはしない。以前のWorker単独結果と、APIによる独立したowned soft-token copyの最大比は同じ値だった。6条件のFP32不合格をcosineやFloat16合格で消さない。

全conditionは公開APIを通し、実入力 / 出力backend、soft token shape `[256/252,512]`、焼き込みposition metadataを確認した。GPU conditionのCPU計算 / input fallback・readback flushはすべて0。診断readbackはAPI結果を受け取ってから行った。これは実encoder featureまでであり、Texture前処理・最終backbone・768次元vector全chainの受け入れではない。[全実測](results/m3-image-encoder-api-real-windows-20261014.json)。

harness sourceは `2539332d74d204668855409fbb945ee72881b59c`。実行は2026-10-13 16:49:26 UTC開始、16:52:12 UTC完了。NUnit6 passed / 6 failed / skip 0 / CLI exit 1。result SHA `bca3bb29163353b5c6c3fe532c5f5d515e9f20423f9701a716a72d51d9c4635b`、CLI receipt SHA `c32b5f7cd416b5b2fb92e7f409cf28d2e023e61c341cb62396f311a8e8a189b6`。Editor6000.3.21f1 / Sentis2.6.1、RTX4070 Ti SUPER / Direct3D12、既存consumer `artifacts/k`、Editor本体PID149180の1つ。公式Test Framework1.6.0 embedded条件を保持する。

## 検出の失敗と実行source

最初のCLIは`NoTestsFound` / 0件 / exit 1であり、実model推論は未実行だった。CLI receipt SHA `f9e852ea26b5b03e13f116a868541141eba1dc6694fb302175be1d2144e7e07b`。compileはerror / warning 0だったが、新規scriptはAssetDatabaseに登録されていなかった。metaのYAML警告を解消し、対象importと通常Refreshも試したが登録は反映されず、根本原因は未確定。

同じfixture sourceを既存の自作test-only `CpuImagePreprocessorIntegrationTests.cs`の別namespace blockへ組み込み、読み込み済み型とassembly `EmbeddingGemma.Editor.Tests`を確認してから全12条件を実行した。Runtimeファイルは変更しない。重複する未importのconsumer fileは除き、元test-only fileをbackupした。後続の定数検査までterminalを確認後、元の自作fileへ復元した。0件runやcompileだけを実model合格に数えない。

## 最終画像参照とのidentityは一致しない

既存の最終画像Python producer `35d357263ae29de4db1ad93d1cac3616ea1e61b4`が束縛するvision reference SHAは `182ba15ea1ff13b4e82d7926d9a743295cbaad019548bd47e2ea2d0c87c675a2`。今回の旧vision sourceとは異なる。tools/でuvによる標準ライブラリだけの分析を行い、旧JSON feature値をlittle-endian float32へ戻したhashと、最終native feature binaryのmanifest hashを比較したところ全6画像で不一致だった。[identity receipt](results/m3-native-vision-context-identity-20261014.json)。

hash不一致は数値差の大きさや許容差超過の証明ではない。同一参照として扱わず、最終chainでは同じproducerのvision / final artifactsを監査する必要がある。今回の旧source結果を最終sourceへ付け替えない。

## 大きいbackbone constantの検査

保存済み音声backboneの元receipt SHA `f79df6aa24fed7f154598a8035bd6cb104dd9d6168db74901be636beb9b9a047`と各model hashを再照合し、harness `431c986a14d8b4be8e43f652bfd3aabb1e409c46`でCPUのみのmetadata検査を行った。NUnit2 pass / failed0 / skip0 / CLI exit0、16:55:37 UTC完了。GPUへschedule / uploadしていない。

- [FP32](results/backbone-constant-fp32-inspection-20261014.json): 最大constantはFloat `(262144,512)`、536870912 bytes（512 MiB）。先のD3D12 logのrequested byte数と一致。次のFloat constantも25165824 bytesで16 MiBを超える。
- [Float16](results/backbone-constant-float16-inspection-20261014.json): 最大constantはShort `(262144,512)`、268435456 bytes（256 MiB）。

これは大きい転送を調べる具体的な対象であり、D3D12終了・Float16 GPU正規化失敗の原因確定や修正ではない。vision側の最大constantはFP32 9437184 bytes / Float16 7741440 bytesで、backboneとは規模が異なる。次は大きいconstantのuploadを小さいTDDから分割できるか、借用model・GPU buffer所有権・FP32 / Shortのbit保存を検証し、その後元の実backbone / 音声全chainへ戻す。PackageCacheを直接変更した成功やD3D11だけの代替を元D3D12の修正として扱わない。

実model FP32、最終画像chain、元音声encoder・GPU安定性、sample / Player / 必須環境・最終空consumerは未達 / 未実行。正式Releaseは作成しない。
