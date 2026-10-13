# M4: 音声対応のTDD計画

更新日: 2026-10-13。**音声end-to-end API / sampleはmainに未実装、実音声encoderのPython数値gateは不合格、Sentis encoder受け入れは未完了**。音声前処理単独はPythonとWindows Sentis CPU / 実GPUで合格済み。[全体計画](multimodal-tdd-plan.md)の工程4として進める。M4実装だけでは公開せず、3モダリティと全必須環境の受け入れを揃える。

A1 / A3の前処理exportと参照・必須Actions gateはTDD実装し、PR #38 / #46をmainへ統合した。[係数・許容差の根拠](m4-audio-preprocessing-validation.md)。35 redから前処理契約を満たし、監査 / 回帰を含め110 passed / skip 0。Windows UnityはCPU / 実GPU各6信号、保存 / 再読込、mask完全一致が合格。最大絶対誤差は両backendとも `0.0038890838623046875`、GPU計算 / 入力fallbackとreadback flushは0件。[実行条件・診断readback・未達範囲](m4-audio-preprocessing-sentis-validation.md)。製品PCM Runtime APIと本番GPU連結、音声encoder以降、他OSの成功とは分ける。

A1 / A4の音声encoder / projector固定profile exportと独立PCM参照・必須Actions gateをTDDで実装した。[実装と参照設定・検証範囲](m4-audio-encoder-validation.md)。encoder20 red、SDPA契約2 red、参照 / CLI8 red、workflow1 redを観測し、最終の関連小さい回帰145 passed / skip 0。実重みActionsとSentis・A5以降は別gateとして残す。

### 直近のTDD順序

1. **A4の原因演算を分離して修正する。** [PR #40](https://github.com/ayutaz/unity-embeddinggemma-2/pull/40)、確認head `4ae52890da50ffc570cba4902d96e95b182cc9c6`は未統合。層内stage欠落 / 誤順序 / hook解放・失敗報告保持を検証する10 red、追加自動定位3 redを経て関連59件green。[実モデルCI 38327706633](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38327706633)は不合格。silenceの公式対eager差0、保存対eager最大差 `0.00013375282287597656`、許容差比 `5.294248580932617`、不一致328要素。[固定headの実測](https://github.com/ayutaz/unity-embeddinggemma-2/blob/4ae52890da50ffc570cba4902d96e95b182cc9c6/docs/results/m4-audio-layer-real-failed-20261013.json)。subsampleは一致、最初の非zero差はlayer 0.feed_forward1、最初の許容差超過はlayer 10。層内traceのfinalは元の失敗graphと一致しており、診断が別経路へ逸れていない。次は公式入力を使うfeed-forward単独のRMSNorm / linear / activation / norm・scale / residualを比較する診断をTDD追加する。stage・入力・hookとstandalone対full trace整合のred → 診断green → 原因演算の再現red → 最小修正 → 元の実重み6信号green。残り5信号は未実行、診断成功をencoder成功と数えず、許容差は緩めない。
2. **A2 / A3の製品Runtime接続。** 前処理graph単独の合格を起点に、PCM境界長・channel / sample rate・NaN / Infinity・空入力・maskのredから入力APIを実装する。GPU tensor受け渡しと各入力経路を確認し、診断fixtureのgreenだけでAPI完了にしない。この契約はencoder修正と独立して準備できる。
3. **A4 → A5の実モデル連結。** encoder数値gateの合格後、Sentis保存 / 再読込・CPU / 実GPU・Float16を照合し、featureをGPUのまま共通本体へ渡して最終768次元を照合する。共通backboneのPR #43は配置・cacheの68件greenと実1024長4条件が合格済みだが、合成featureによるbackbone単独の結果。後続の固定Git Runtime更新consumerでも4条件合格したが、公式prompt構築・実音声feature接続と最終candidateの空consumerは別gateとして扱う。[Git更新のreceipt](results/multimodal-git-runtime-upgrade-windows-20261013.json)。[固定headのbackbone結果](https://github.com/ayutaz/unity-embeddinggemma-2/blob/21e598c5b53d5ad73557ee6d606b12770254d73b/docs/results/multimodal-backbone-sentis-windows-20261013.json)。
4. **A6 / A7のsampleとPlayer。** 検索・再生・明示収録・停止 / 解放を下表のredから実装し、固定PCMの精度とマイク入力動作を別に受け入れる。テキスト・画像回帰と必須実機へ渡す。

過去の368要素不一致の実測も保持する。新runの328要素への変化は原因解消を意味しない。音声前処理の合格、層内診断59件green、実encoder不合格を個別に記録する。

[段階別実モデルCI 38324421389](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38324421389)は不合格で完了。subsampleは完全一致、`layer_0`から小さい差があり、許容差を最初に超えるのは`layer_10`。診断graphと元の不合格graphの最終値は完全一致した。層内診断は小さいTDDで追加済み。次は実重みActionsで原因演算を特定し、元の固定数値条件へ戻す。[全stageと未検証ケース](m4-audio-encoder-validation.md)。

Feed-forward診断は追加16 red → 16 green、余分なexportを防ぐ追加1 red → 関連76 greenまで実装した。[TDDの証拠](results/m4-audio-feed-forward-tdd-20261013.json)。公式入力・各演算の値と単独対full stage一致を比較する。実重みでの原因演算と元の6信号合格は未確認であり、次のActionsで判定する。PR #43は全8 CI成功後にmainへ統合済み。合成featureのbackbone単独と、実音声encoder連結の未完了を分ける。

## 範囲と参照固定

A1 / A3のPython前処理exportは[PR #38](https://github.com/ayutaz/unity-embeddinggemma-2/pull/38)でTDD実装し、mainへ統合済み。head `0424734291dc4cb7606b8bae53086f4911252e94`の[CI 38313045325](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38313045325)全8 job成功、公式前処理対保存graphの6ケースで最大絶対誤差 `0.0038890838623046875`。このproducer報告の`audio_preprocessing_only` / `unity_validated=false` / `encoders_validated=false`は当時の実行範囲として保持する。後続のSentis前処理受け入れは上記の別receiptへ記録する。次は入力契約・Runtime接続、A4の音声encoder exportと独立参照照合を進め、A5〜A7へ接続する。

PCM波形 → GPU上のSTFT / mel / log等の公式前処理 → 音声encoderのsoft token → 共通本体 → 768次元を実装する。最初はbatch 1と固定長profileを扱い、音声の最小 / 最大長、padding / mask、長さ超過時の扱いを契約にする。長音声のchunk化は参照のtoken配置・集約を確認してから追加し、独自のchunk平均を参照互換としない。

固定revisionの公式processor・実装からsample rate、channel処理 / resample、window係数とperiodicity、center / pad、frame数、FFT、amplitudeかpowerか、mel scale / normalization / 範囲、floor / log / affine処理、paddingとencoder出力長、prompt / audio特殊tokenを確認する。既存調査の16kHz / window320 / hop160 / FFT512 / mel128 / floor0.001だけでは参照一致を保証しない。公式参照確認後に全係数・許容差・profileをfixtureへ固定する。

無音、impulse、sine、channel差、hop / window境界±1 sample、極短・上限境界を前処理fixtureとする。意味検索には権利・同意を確認した固定音声と日英text queryを別途使い、hash / 出典 / 長さ / sample rateを記録する。fixtureはユーザー入力の自動収集から作らない。

## 実装順とRed / Green

| ID / 依存 | 先に観測するRed | 最小実装 | Greenと実モデルの受け入れ |
| --- | --- | --- | --- |
| A1 参照・C2 | processor差、frame / mask / soft token数、無音floor、sample rate不一致を参照監査が見逃す | waveform・前処理・encoder・最終vectorの参照CLI、profile / manifest | 小さいfixture green後、Actions実モデル参照を固定。中間tensor / 最終score許容差を実装前に決める |
| A2 入力・A1 | channel順・PCM振幅・sample rate・trim / padが異なる。NaN / Infinity・空入力・対応外長を誤って許可する | PCM入力契約、参照通りの必要なdownmix / resample / mask。未対応は明示拒否 | boundary / 不正入力テストgreen。decode / 収録のCPU処理とGPU推論の境界を明示し、採用各入力経路を参照照合 |
| A3 GPU前処理・A1 / A2 | window / center / pad / FFT振幅 / mel / logの各段階が固定Pythonと不一致 | Sentisの対応演算でSTFT等を構成し、前処理をGPU上で実行 | silence / impulse / sineと実音声の各段tensor一致、actual GPU、入力upload後の中間CPU readbackなし |
| A4 encoder export・A1 | padding / mask、出力長 / token順が実Pythonと不一致。必要演算の小さい互換性テストも先に作る | 音声encoderを対応演算へ展開して`.pt2` / `.sentis`化、採用Float16重み形式 | Actionsでexport数値照合、Unity実モデルCPU / GPUでfeature照合。import成功・tiny greenを実encoder合格にしない |
| A5 共通本体・A3 / A4 / C3 | audio placeholder数 / feature順、prompt、mask / projection / pooling不一致、token上限超過がred | GPU featureを共通本体へ渡す。仮API名 `AudioEmbedder`、長さと所有権を契約化 | 実モデルfp32 cosine >= 0.999 / 採用Float16重み >= 0.99、有限768次元・単位長。text128 / M3の回帰。無音を勝手なzero vectorへ置き換えない |
| A6 検索・sample・A5 / C5 | text→audio全順位 / score・同点ID順が不一致。欠落モデル・停止 / 再操作・入力不正の状態契約がred | 固定音声の事前embedding、text query・結果と再生、明示開始 / 停止の音声入力 | 固定query全順位一致、sampleで準備→検索→結果音声再生→停止 / 解放・復帰を実操作。任意発話の意味品質は別評価 |
| A7 収録・Player・A6 / C4 / C6 | permission拒否、deviceなし、収録終了 / 中断、profile不足、モデル破損、GPU fallback・非zero終了を誤ってpassにする | マイク利用は明示開始 / 停止・権限失敗 / resource解放を実装。bundle監査・Player測定 / 終了監査を接続 | 固定PCMで実GPU精度、マイクは別の入力動作検証。Windows IL2CPP / Release / High stripping、独立2起動実終了0。モバイルの権限 / suspendも実機確認 |

マイクは端末由来の音声がCPUメモリへ入る入力境界として扱う。最終的な受け入れで「PCM取得 / upload」と「波形以後の前処理 / encoder / 共通本体」を分け、後者をGPUで処理する。モデル入力は一時的に保持し、利用者の録音・発話を自動保存・送信・学習収集しない。

A3 / A4の非対応演算・shape・OOMは失敗を保存する。GPU前処理をCPUへ移すことで条件達成とはしない。必要な標準演算分解やONNX再exportは、小さい失敗テストから同じ実モデル参照へ進め、native pluginを導入しない。

## 測定・完了条件

- A1〜A7の意図したred / green、固定係数 / mask / profile、source・model / processor / graph hash、requested / actual backendを保存する。
- 収録 / decode / upload、STFT / mel、encoder、共通本体、load / 初回 / warm反復、再生 / 切替 / Disposeを分け、時間・メモリと測定APIを記録する。
- 実音声の参照embedding、text→audio全順位、text / image回帰、mixed index、空Git consumerのsample検索 / 再生 / 明示収録 / 停止が合格する。固定PCMの精度成功をマイク入力動作の成功へ流用しない。
- Windowsの結果からmacOS / iOS / Androidや音声権限・GPU対応を推定しない。未実行 / 失敗は[全体公開条件](multimodal-tdd-plan.md)へ残し、実機のメモリ収支・正常終了まで確認する。

PRはA1 / A4の参照・export、A2 / A3の入力・GPU前処理、A5 / A6のAPI・検索 / 再生、A7の明示収録・Player / platform接続に分ける。M3と共通基盤を再利用し、変更範囲に応じて必要なテキスト・画像回帰を行う。
