# M4: 音声対応のTDD計画

更新日: 2026-10-13。**音声end-to-end API / sampleはmainに未実装、実音声encoder / Unityの合格証拠は未取得**。共通backboneの一部はPR #34で実装済みだが、合成特徴量によるPython照合は音声encoderの受け入れを満たさない。依存・実行場所・公開条件は[全体計画](multimodal-tdd-plan.md)、既存仕様調査は[技術調査](technical-approach.md)。M4の実装完了だけで公開せず、3モダリティと全必須環境の検証を揃える。

A1 / A3の固定profile音声前処理exportと参照・必須Actions gateをTDDで開始した。[係数・許容差の根拠と検証範囲](m4-audio-preprocessing-validation.md)。35 redから前処理契約が合格し、監査 / 回帰を含め110 passed / skip 0。Unity実GPUとaudio encoder以降は未検証。

### 直近のTDD順序

1. A4の保存graph数値差を解消する。[PR #40](https://github.com/ayutaz/unity-embeddinggemma-2/pull/40)の確認head `c4676f7b38f01ce98a9dedca87fb450febea20b9`は未統合。段階別診断をTDDで追加して関連46件greenとなったが、[実モデル診断 CI 38324421389](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38324421389)は不合格。silenceの公式対eager差0、保存graph対eager最大差 `0.00021576881408691406`、許容差比 `9.733773231506348`、不一致368要素。subsampleは一致し、layer 0から微差、layer 10で初めて許容差超過となった。[固定headの診断結果](https://github.com/ayutaz/unity-embeddinggemma-2/blob/c4676f7b38f01ce98a9dedca87fb450febea20b9/docs/results/m4-audio-stages-real-failed-20261013.json)。次は層内のstage欠落 / 誤順序 / hook解放漏れ / 元の失敗消失を拒否するred → 層内診断green → 原因演算の小さい再現red → 最小修正 → 固定実重み・6信号・mask profileのgreenを得る。残り5信号は未実行。最新headのCI 38325813586は今回確認時点で実行中。診断の成功はencoderの成功ではなく、許容差を緩めない。
2. A2のPCM入力契約とA3の前処理graphをSentisへ接続する。境界長・mask・無音・channel / sample rateのredから実装し、各段の実CPU / GPU結果を確認する。
3. A4のencoderをSentisで保存 / 再読込し、CPU / 実GPUとFloat16を照合する。A5で実音声featureをGPUのまま共通本体へ渡し、最終768次元を固定参照と照合する。共通本体APIのPR #43には小さい実GPUと配置演算 / cache修正を含む68件のgreenがあるが、修正後の実1024長graph・実音声featureの照合と新Runtime assemblyの固定Git導入は別の未完了条件として確認する。
4. A6 / A7の検索・再生・明示収録・停止 / 解放をTDDで実装し、固定PCMの精度とマイク入力動作を別に受け入れる。テキスト・画像回帰と必須実機へ渡す。

音声前処理exportの合格を音声encoderの合格に含めない。PR #40の小さい契約greenと実モデル不合格も別に保持する。

## 範囲と参照固定

A1 / A3のPython前処理exportは[PR #38](https://github.com/ayutaz/unity-embeddinggemma-2/pull/38)でTDD実装し、mainへ統合済み。head `0424734291dc4cb7606b8bae53086f4911252e94`の[CI 38313045325](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38313045325)全8 job成功、公式前処理対保存graphの6ケースで最大絶対誤差 `0.0038890838623046875`。`audio_preprocessing_only` / `unity_validated=false` / `encoders_validated=false`であり、A3のSentis GPU合格ではない。次は入力契約、前処理graphのSentis実測、A4の音声encoder exportと独立参照照合を進め、A5〜A7へ接続する。

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
