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

## 次の実装と実機確認

1. Player側でbundleを読み、起動時に実際のファイルhashとcode commitを記録する。Androidのjar内StreamingAssetsは通常のFile読み込みと分け、事前配置 / persistentDataPathへの展開をTDDで実装・実機確認する。
2. NUnit / UnityEditorに依存しない実行componentを用意し、15ケースのtoken ID / mask完全一致、4条件の埋め込み、6文書 / 4queryの全順位を検証する。Editorで使った古い結果をPlayer成功にしない。
3. requested / actual backendを記録し、GPU非対応やOOMを失敗にする。GPUをCPUへ黙って切り替えない。
4. load、tokenizer準備、初回推論、warmup後の反復推論、解放を測る。OS、端末、Unity、GPU / graphics API、build backend、stripping、メモリcounterの取得可否と観測範囲を記録する。
5. Windows Playerからbuild / 起動 / 数値 / 終了 / 再起動を確認し、macOS Editor / iOS / Androidへ同じ参照・測定を渡す。buildだけでは実機GPU成功ではない。

fixtureのred確認後、bundle / stage / Git consumer / packageの対象70テストが合格した。
既存の監査済み実モデルから約1.68GBのbundleを作成し、全5ファイルのcopy前後SHA-256が元の証跡と一致した。モデルの再download・変換は行っていない。
この実行は既存モデル・filesystem cacheを使用した単回の配置とhash監査であり、Player build / 起動、macOS / iOS / Androidは未実行。
Unity用Secrets / runner / 端末が必要な経路と、Python・静的監査で進められる準備を区別する。
