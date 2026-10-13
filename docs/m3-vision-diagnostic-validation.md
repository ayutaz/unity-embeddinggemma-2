# M3: 画像FP32不一致の層別診断

2026-10-13。画像encoderの既知のFP32不合格を解消するため、独立したnative stage参照とSentis診断を追加した。診断の合格と、元の3 profile / 12条件の合格は別に判定する。正式Release / tagは作成しない。

## 固定する経路

- 公式vision model / projectorへhookを付け、patch embeddings、rotary cos / sin、各encoder layer、pooling / standardization後、pre-projection norm、projected featuresをcloneして保存する。export wrapperの出力を期待値には使わない。自分のhookだけを成功 / 失敗時に除去する。
- 最初のprofileの両画像を1つの診断graphへ通す。残るprofileを合格と数えず、最初のprofileも実モデル数値修正後に元の受け入れへ戻す。
- 保存診断graphの最終値が元の保存graphと完全一致しない場合、Python診断を成功にしない。Unityでも以前不合格だった保存Sentis modelのhashを照合し、両画像の最終値を診断graphと比較する。
- 原因判定は、Pythonの元経路一致、Sentisの元経路一致、元のFP32不合格の再現が揃った場合だけ有効とする。微差の最初のstageと、固定許容差を超える最初のstageを別に記録する。
- `rtol=1e-4` / `atol=1e-5`を維持。rotaryなどの定数出力がCPUに存在することと、実際の計算 / 入力CPU fallbackを区別する。動的出力はrequested backendと一致し、GPUでは計算 / 入力fallback・readback flushを許可しない。

## TDDとPython実モデル

診断12 assertion red → 12 green。CLI / routingは未実装の3 API失敗から、既存prepareを含む13 green。両画像の省略を3 assertion redで検出し、修正後は関連26 green。rotary出力の欠落を2 assertion redから修正し、診断13 green。[Python TDD receipt](results/m3-vision-diagnostic-python-tdd-20261013.json)。

[CI 38344076080](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38344076080)は全8チェック成功。診断producer sourceはCI merge ref `736a4684be754563bbb953a67c2023d688c4a312`、PR head `5760d7b`とは別。参照生成と重み / 変換はActionsで実行し、ローカルで実モデルPythonを再実行していない。

Unityの元経路一致 / 不合格再現監査は12 assertion red → 12 green。実モデルharnessはcompile Error 0 / Warning 0。[Unity監査TDD receipt](results/m3-vision-diagnostic-unity-tdd-20261013.json)。小さい監査の成功を実モデルの原因特定やFP32合格とは数えない。

## 取得と実測

artifact全体には不要な3 profileの再取得も含まれるため、HTTP Rangeのseek / block境界・不完全応答・不要モデル省略・パス監査を4 assertion red → 4 greenで追加した。選んだ`diagnostics/`だけを取得し、全file hash・固定source / revisionを照合してからUnityへ渡す。秘密情報や署名URLは結果へ保存しない。

実CPU / GPUの層診断は参照取得後に実行する。初回import / 保存を1回にし、同じ保存modelを各backendで再読込する。診断出力のreadbackは実行完了後の測定境界であり、製品GPU前処理からencoder / backboneへの連結を証明するものではない。

初回の選択取得は、native参照の取得後、graphの369098752 bytesまで進んだ時点で期限付きURLのHTTP 403により終了した。部分graphは退避して保全し、取得済み参照をZIPのsize / CRCで再利用する追加1 assertion red → 5 greenを確認した。最終信頼判定はCRCだけではなく、固定source / revisionとproducer manifestの全SHA-256照合で行う。URLは403時に更新し、GitHub認証情報・署名URLを結果へ保存しない。取得の再開は明確なterminal failure後に行い、観測timeoutやUnity再起動とは分ける。再取得は完了し、49選択fileを全hash照合してUnityへ渡した。再取得分696421292 bytesは失敗した初回分を含む総転送量ではない。

### 実Sentisの失敗と参照文脈

固定harness head `d51262632b0d0369fe62719dff743e6ecd990a46`のsourceを検証用consumerのEditor assemblyへ配置した条件で測定した。product Runtimeのoverlayはない。初回はBurstのxxHash3 function pointerのcompile失敗でTorch importが終了し、CPU / GPU数値推論は未実行。未保存scene 0を確認し、旧Editor終了後に6000.3.21f1を1つだけ起動した。回復後は同じgraphのimport / 保存が成功した。これは性能改善や数値修正ではない。

2026-10-13 04:56:26 UTCの実CPU / GPUCompute診断はcompile成功、0 passed / 2 failed / 0 skipped。両backendのcase_0でtrace最終値は元保存Sentisと完全一致したが、`New native final must fit the original pinned reference`に失敗した。21 stageを読み出した途中結果があり、case_1は未実行。途中の`localization_validated=true`はこの後段gateの前に設定されていたため、診断受け入れ・原因確定には使用しない。元のFP32不合格は維持する。

同じproducerの親`vision-reference.json`だけを追加で範囲取得し、親manifestのSHA-256と固定sourceを照合した。両画像とも、親の公式最終値と診断hookの最終値は完全一致した。一方、元の固定参照との不一致は画像0で598要素 / 最大許容差比5.768531797804444、画像1で51要素 / 比2.58704231960917。入力pixel hash、model / config / processor hash、ライブラリ版は一致した。新旧producerのvision forward・loader・uv lockには変更がなく、hook追加がこの差を作った証拠はない。CPU演算環境等の原因は未確定。

### 参照判定修正のTDD

不一致native参照でも、2つのSentis最終値が一致するだけで診断成功を出す問題を追加1 assertion redで再現した（既存・対照14 passed）。固定許容差でnative参照も照合する最小修正後、15 passed / failed 0 / skipped 0。harnessは参照の誤差を保存し、native gateと全stageの完了前にはlocalization成功を設定しない。小さい判定契約のgreenで実Sentis診断を合格へ変更しない。修正後の実モデルharnessは未実行。

新しい参照にはCPU capability、intra / inter-op threads、Float32 matmul precision、MKL / MKLDNN・deterministic設定、Torch build情報を記録する。参照・診断metadataの欠落と設定変更を3 assertion redで確認し、最小実装後に関連27 passed。数学設定・許容差・元参照は変更しない。[失敗・比較・TDD receipt](results/m3-vision-reference-context-windows-20261013.json)。

次はActions生成の演算環境を確認し、固定参照との再現条件を小さい比較で切り分ける。参照文脈が揃ってからstage / 演算のredへ進み、元の3 profile / 全12条件に戻す。過去の失敗や未実行を上書きしない。

数値不合格の原因修正、元の全12条件、実画像の最終768次元API / sample、GPU profiler・資源 / 性能、Player / 必須他OS、最終空consumerは未完了。追加診断だけでこれらを合格にしない。
