# Player / 他環境の検証準備

2026-10-10。M2のmacOS Editor / iOS / Android実測とWindows Player回帰の準備を進める。
受け入れ条件とリリースの依存は [詳細実行計画](m2-release-plan.md) に従う。

## 実装済み: hashを照合したモデルと固定参照のbundle

`tools/embeddinggemma_tools/player.py` はEditorで準備したFP32 / Float16の `.sentis` とtokenizer、監査済みM1 / 検索参照を、空の出力先へまとめる。
Unityの再import・変換・量子化・GPU推論を行う処理ではない。出力の `unity_executed` / `gpu_verified` は常にfalse。

モデル元のSHA-256、model / search生成commit、固定model revision、Unity準備version、全ファイルの長さとSHA-256を照合する。
M1と検索のモデル設定・Python依存versionを照合し、固定shapeと有限・単位長の参照を確認する。
copy後もhashを確認し、copy中の変更や失敗で成功receiptを残さない。既存のbundleを上書きしない。

### 実行

既存consumer `artifacts/c` のprepared modelsと監査結果を使用する例。`run-001` は新しい空の出力先へ変更する。
Pythonは `tools/` のuvを使用する。大きいcopy / 変換は実行環境で必要な時だけ行い、生成物をGitに含めない。

```powershell
cd tools
uv run --locked python -m embeddinggemma_tools.player --prepared ../artifacts/c/Assets/StreamingAssets/EmbeddingGemmaTextSearch --reference ../artifacts/c/artifacts/m1 --audit ../artifacts/c/artifacts/m1-stage.json --output ../artifacts/player-bundle/run-001
```

出力:

```text
run-001/
  model-fp32.sentis
  model-float16.sentis
  tokenizer.json
  reference.json
  search-reference.json
  bundle.json
```

モデル元の `.pt2` やネイティブプラグインはbundleに含めない。
`bundle.json` のsuccessは配置・hash監査が完了した意味であり、モデルのPlayer / GPU実行やM2合格を意味しない。
出力に部分ファイルが残ってもreceiptがfalseなら使用しない。失敗原因を確認し、新しい空出力先で必要な処理を行う。

## 実装済み: Playerから利用する検証ロジック

`Assets/Validation/Runtime/` に検証専用assemblyを置いた。配布UPMのRuntimeとは別で、NUnit / UnityEditorへの参照を持たない。opt-inのPlayer起動componentとWindows build helperを接続した。

- `PlayerBundleLoader.Load(directory)` はreceiptの固定revision・生成commit・Unity準備version・固定5ファイルを確認し、毎回全ファイルの完全SHA-256を計算する。同じサイズ・mtimeでも内容が違えば失敗する。hashを確認した同じfile handleでtokenizerと参照を読む。
- `PlayerValidation.ParseReferences` はM1 15ケース、検索6文書 / 4queryのshape・有限単位vector・prompt・全順位・依存metadataを確認する。model / search生成commitはbundleに照合する。
- `CheckAllTokens` はM1と検索を合わせた全25入力で、固定参照とtoken ID / maskの全128要素を完全一致で確認する。実行側は実際の`TextTokenizer.Encode`を渡す。
- `RunCondition` は明示したprecision / backendを実行し、15ケース・検索の全順位とscore・追加warmup後3回の推論・解放を記録する。FP32 cosine >= 0.999、Float16重み >= 0.99、有限768次元・単位長を要求する。requested / actual backendが違えば失敗する。
- `AllConditionsPassed` はCPU / GPUCompute × FP32 / Float16の4条件が重複なく揃い、必要件数・精度・解放が合格した場合だけtrueを返す。欠落・重複・CPU代替・エラーを成功にしない。

uloop-cliで実装前のredを20件、9件、追加11件と段階的に観測し、最終的にUnity Editorで40 passed / failed・skipped・inconclusive 0を確認した。[TDDの記録](results/m2-player-contracts-windows-20261010.json)にsource hashと実行条件を記載する。
テストは小さい合成参照 / providerを使用しており、実モデル推論、実際のGPU、Player build / 起動の証拠ではない。既存のSentis実モデル結果と分けて扱う。

現時点のloaderは通常のfilesystem専用。Androidのjar内StreamingAssetsやURLは明示的に拒否する。[Unity公式のStreamingAssets仕様](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Application-streamingAssetsPath.html)に沿った展開adapterと実機確認は次の作業に残す。

## Windows検証Playerのbuildと起動

Unity 6000.3.16f1 / Sentis 2.6.1のプロジェクトでWindows 64-bitを選択し、Playを停止してsceneを保存する。必要なcompileと契約テストを先に実行する。`Assets/Validation/` は検証用で、UPM Runtimeへ配布しない。

Editorから `EmbeddingGemma.Validation.Editor.ValidationPlayerBuild.BuildWindows("<新しい空directory>/Validation.exe", "<ソースの40桁commit>")` を呼ぶ。既存出力を上書きせず、生成した検証sceneだけをbuildし、元のsceneを復元して一時markerを削除する。`build.json` はbuild結果、時間、errors / warnings、Mono / IL2CPP、stripping、RuntimeソースのLF正規化SHA-256を記録する。buildを呼ぶ前にconsumerのソースと指定commitが一致することを確認する。

検証PlayerにはResourcesのprovenance markerを含める。markerがない通常PlayerやEditorでは自動実行しない。監査済みbundleを通常filesystemに配置して、毎回新しいresult path / run IDを指定する。

```powershell
& '<build directory>/Validation.exe' -batchmode -force-d3d12 -logFile '<run directory>/player.log' --embeddinggemma-bundle '<bundle directory>' --embeddinggemma-output '<run directory>/results.json' --embeddinggemma-run-id '<32桁の小文字hex>'
```

`results.json` の既存ファイルを上書きしない。実行途中はsuccess=falseの完成したJSONを保存し、全25入力のtoken一致と実モデル4条件の合格後だけsuccess=trueにする。Windowsの短い読み取りロックに対する置換の再試行は最大1秒で、継続する書き込み失敗を合格にしない。終了code 0とcompletedの結果、run ID、buildのソースhashを併せて確認する。

時間はモデルdeserialize、worker / tokenizer準備、初回推論、追加warmup後3回のquery推論、解放を分ける。推論の時計にはprompt / tokenization / schedule / 最終vector readbackを含む。メモリはapplication全体のstage sampleであり、peakやモデル専用使用量ではない。GPU使用量はunknown、取得できないcounterはnullとする。成功後はprocess終了を確認し、同じbinaryを新しい出力先で再起動して確認する。

## 残る実機確認

1. Androidのjar内StreamingAssetsは事前配置 / persistentDataPathへの展開adapterとして別にTDDと実機確認を行う。現在の起動経路は通常filesystem用で、Android対応の成功には数えない。
2. macOS Editor / iOS / Androidで実際のtokenizer / Sentis providerを実行する。15ケースのtoken ID / mask完全一致、4条件の埋め込み、6文書 / 4queryの全順位を実環境で確認し、Windowsの結果を他環境の成功にしない。
3. requested / actual backendを記録し、GPU非対応やOOMを失敗にする。GPUをCPUへ黙って切り替えない。
4. load、tokenizer準備、初回推論、warmup後の反復推論、解放を測る。OS、端末、Unity、GPU / graphics API、build backend、stripping、メモリcounterの取得可否と観測範囲を記録する。
5. macOS Editor / iOS / Androidへ同じ参照・測定を渡す。buildだけでは実機GPU成功ではない。

fixtureのred確認後、bundle / stage / Git consumer / packageの対象70テストが合格した。
既存の監査済み実モデルから約1.68GBのbundleを作成し、全5ファイルのcopy前後SHA-256が元の証跡と一致した。モデルの再download・変換は行っていない。
このbundle作成自体は既存モデル・filesystem cacheを使用した単回の配置とhash監査であり、Player build / GPU成功の証拠ではない。
Unity用Secrets / runner / 端末が必要な経路と、Python・静的監査で進められる準備を区別する。

## Windows Player実測: 2026-10-10

[機械監査した結果](results/m2-player-windows-20261010.json)。buildのソースcommitは `dfdcc9cb440cceca0380022a528a54f27e709cfe`。既存のローカルfile依存consumer、Unity 6000.3.16f1 / Sentis 2.6.1 / Windows 11 / RTX 4070 Ti SUPER / Direct3D12、Development / Mono2x / stripping Disabledで実行した。

- 実装前にprotocol 17件、build 5件の失敗を確認した。scene復元の実際の失敗1件も修正した。結果保存の競合は追加テストの2件でredを観測し、最終的に65 passed / failed・skipped・inconclusive 0だった。この契約結果自体は実GPU成功ではない。
- 最初のPlayerはcommit `84f7426` で終了code 1。結果ファイルの置換が失敗し、モデル監査と推論には進まなかった。短いreader lockを再現して最大1秒の置換再試行を追加した。元の競合processは特定していない。
- 修正後の同じbinaryを2つの独立したrun IDで起動し、終了を確認してから再起動した。両方とも終了code 0、実モデル4条件合格、GPU非代替、25入力のtoken ID / mask完全一致、M1 15ケース、検索6文書 / 4queryの全順位・score合格、worker解放を確認した。
- 実モデルの再download / 変換は行わず、既存の監査済みbundleを使った。全5ファイルを起動ごとに完全SHA-256照合した。最低cosineは全条件で `0.999999999998` 以上、最大検索score差は `2.43e-7` 未満だった。

| 重み / backend | 初回推論 ms: run 1 / 2 | warmup後query平均 ms: run 1 / 2 |
| --- | ---: | ---: |
| FP32 CPU | 577.2 / 574.3 | 509.9 / 486.2 |
| FP32 GPUCompute | 4631.6 / 1661.6 | 55.0 / 36.7 |
| Float16 CPU | 1082.0 / 861.0 | 794.6 / 802.7 |
| Float16 GPUCompute | 667.1 / 562.9 | 40.5 / 43.6 |

各平均は同じqueryの3回分。2回の起動ともimport / shader / filesystem cacheを再利用しており、cold-cache計測や統計的な速度比較ではない。hash監査は約22秒、tokenizer準備と全token照合は約4秒だった。ロード・worker準備・解放・メモリのstage sampleはJSONに記録した。

最初のbuildは170.3秒、修正後は同じEditor / build cacheで11.2秒。両方ともerrors 0 / warnings 485で、Sentis ConvGeneric shader variantの警告を含む。警告ゼロのbuildではない。終了時のUnity `MemoryLeaks` 診断にも約1.10MBの値が残るため、allocation問題を解消済みとはしない。workerのDispose完了と、全native memoryが解放済みであることは別の条件として扱う。

Windows IL2CPP / release stripping、Git URL consumerの全動作、macOS Editor / iOS / Androidは未実行。これらと安定性の追跡を残しており、Windowsの合格だけでM2全体や正式リリースを完了としない。
