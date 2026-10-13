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

初回の選択取得は、native参照の取得後、graphの369098752 bytesまで進んだ時点で期限付きURLのHTTP 403により終了した。部分graphは退避して保全し、取得済み参照をZIPのsize / CRCで再利用する追加1 assertion red → 5 greenを確認した。最終信頼判定はCRCだけではなく、固定source / revisionとproducer manifestの全SHA-256照合で行う。URLは403時に更新し、GitHub認証情報・署名URLを結果へ保存しない。取得の再開は明確なterminal failure後に行い、観測timeoutやUnity再起動とは分ける。実Sentis層診断の結果はまだない。

数値不合格の原因修正、元の全12条件、実画像の最終768次元API / sample、GPU profiler・資源 / 性能、Player / 必須他OS、最終空consumerは未完了。追加診断だけでこれらを合格にしない。
