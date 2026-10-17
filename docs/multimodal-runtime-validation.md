# 共通マルチモーダル本体のRuntime契約

> 設計・検証の履歴です。本文の状態・実行中・次の作業は記載当時のものです。現在の対応範囲は[検証状況](status.md)、残作業は[ロードマップ](roadmap.md)を参照してください。固定sourceの失敗・未実行も証拠として保持します。

更新日: 2026-10-13。[全体TDD計画](multimodal-tdd-plan.md)のC3 / C4を進める。`MultimodalBackboneProfile`と`MultimodalBackbone`をUPM Runtimeへ追加し、合成project済み特徴量を使う実text重み・1024長backboneのSentis受け入れを確認した。**画像 / 音声encoder、公式prompt構築、GPU前処理から最終768次元までのend-to-end照合は未完了。** 既存consumerを固定Git Runtimeへ更新した4条件も合格した。最終candidateの空consumerと実encoderのend-to-endは別の残作業とする。以下の初回TDD / import失敗の履歴と、末尾の修正後実モデル4条件の合格を分けて記録する。

## APIと所有権

- 固定batch 1、長さ2〜8192、image / audio feature数とplaceholder ID、text vocabulary / pad IDをprofileで指定する。feature幅512、最終embedding幅768は固定。
- int32 IDs / binary maskの長さ、非空mask、通常IDの範囲、masked placeholder、image / audio slot数を推論前に拒否する。feature順の数値処理は保存graphが担当し、Runtimeはtoken順とfeature row順を変更しない。
- Modelはstaticな4入力 `input_ids` / `attention_mask` / `image_features` / `audio_features`と1出力を要求する。名前、型、shapeをprofileへ照合する。
- 呼び出し側がModelとproject済みfeature Tensorを所有する。backboneはWorkerと各呼び出しのIDs / maskのみを所有し、borrowed featureをDispose / CPU downloadしない。
- 非空featureは既に指定backendへ置かれたlive Tensorでなければならない。CPU / GPUComputeを明示し、GPUPixelや非対応端末を拒否する。GPUへ自動uploadするfallback APIにはしない。
- 最終outputのshape / actual backend、全finite、L2 normを検査する。最終vectorだけを独立したCPU配列へコピーし、Disposeは冪等。Dispose後の推論は拒否する。

GPU featureの全値をhost検査するためのreadbackは行わない。呼び出し側でencoder / artifactのfinite契約を検証し、Runtimeは最終vectorのfinite条件も確認する。画像 / 音声なしのzero-feature profileは型として許容するが、実profileのimport・zero-size tensorを含む実GPU受け入れは別途必要。

## TDDと実行範囲

基準mainは `e35470212f616b020c0b150a4c515a007f0b937f`。未コミットの未実装stubと先に書いたテストを、既存consumer `artifacts/k`のAssets内test assemblyへ配置して実行した。Runtime package / vendor cacheを上書きしていない。配布assemblyへの新規Git導入の証拠とは区別する。

- compile成功後、初回は22件中21 failed / 1 passed / skip 0。profile / layout / model契約、不正output、GPUでのCPU feature拒否、所有権・結果の数値について意図したassertion失敗を確認。
- 最小実装後は22 passed / skip 0。
- 小さい実GPUの追加受け入れを含め23 passed / skip 0。RTX 4070 Ti SUPER / Direct3D12、Unity 6000.3.21f1 / Sentis 2.6.1。GPU featureのbackend維持、実worker / output、CPU layer計算とreadback flushなし、Dispose後もcallerのTensorがliveであることを確認した。小さいgraphの結果であり、実model / 1024長の合格ではない。
- 破棄featureを渡す経路で、APIのArgumentExceptionではなくSentis内部assertionが発生する追加1 redを確認。dataの生存をbackend参照より先に検査する最小修正を追加した。
- 最終は新共通API 24件と既存UPM契約38件の計62 passed / skip 0。compile成功、上記の実GPU受け入れを含め合格。[source hashと段階別結果](results/multimodal-runtime-windows-20261013.json)。
- `tools/`からuvで配布監査・別consumer manifest準備が成功。新source / meta / GUID・Runtime依存とモデル混入を監査した。このmanifest準備はUnity起動・Git解決の証拠ではない。

元のCLI呼び出しに存在しない`--timeout`を渡した引数エラーは、テストが未実行であるためTDD redへ含めない。CLIが案内した`--timeout-seconds`で実行した。Editor本体は既存PID 78588の1つを再利用し、重いmodel実行・新規Editor起動は行っていない。

## 次の受け入れ

### 実backboneのimport不合格と小さい再現

main `e35470212f616b020c0b150a4c515a007f0b937f`の[Actions 38320110239](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38320110239)から、実text重み・1024長・画像280 / 音声280のbackbone artifactを取得した。全ファイルのsize / SHA-256、固定revision、別取得したconfigの元hash一致を監査した。入力特徴量は合成値であり、実encoder出力ではない。

WindowsのCPU / GPUCompute × FP32 / Float16の4条件は**Sentis importで不合格**。`Where.InferPartial`で `ValueError: broadcast dims must be equal or 1`が発生し、ModelAssetが生成されなかった。保存 / 再読込・CPU / GPU数値照合は未実行。CLIの観測は305秒で終了したが、同じEditor / operationの保存報告は最後まで追跡し、2026-10-13 00:01:56 UTCに4条件の不合格完了を確認した。観測timeoutによる重複起動・再試行はしていない。

このimportの問題を小さい固定graphで再現した。Pythonの保存graph監査で1件のredを確認し、特徴量の多次元advanced indexをflat `index_select`と明示的な `[1, length, width]`へのreshapeに置き換えた。非連続 / 並び変更placeholder、feature値変更、padding、ゼロfeatureを含む既存数値契約とtext export / model回帰は25 passed / skip 0。固定shapeのbatch 1という元の対応範囲を維持し、独立masked-scatter参照は変更していない。

Unityでは旧graphの小さいimport 2条件と、同じimport失敗を4回繰り返してしまうcache契約1件の計3 red（他3件green）を確認した。旧 / 新graphの独立Python参照JSONはhash完全一致。flat Gatherへの修正後、小さい保存 / 再読込graphのCPU / 実GPUで2ケースの全24要素が許容差内、GPUのCPU計算fallback / readback flushなし。失敗も保持するLazy cacheの最小修正後は、同じ監査済みsourceを一度だけprepareし、新しいauditではretryできる。最終は追加6件と既存62件の**68 passed / skip 0**、compileも成功した。[入力・source hash・失敗とgreenの範囲](results/multimodal-placement-windows-20261013.json)。

最初の実graphは失敗する同一importを4条件で繰り返した。cache修正はprepare回数を4→1にする契約の証拠であり、実モデルの速度改善を測定した結果ではない。初回不合格の生成assetはartifactへ退避し、元の監査済みgraphと報告を保持した。新しい実backboneの生成 / 独立Python照合はActionsで行い、そのartifactで実Sentisを再受け入れする。小さいgraphのgreenは実1024長graph、新Runtime assemblyのGit導入、実encoder接続のgreenではない。

1. 上記62件の契約greenを起点に、small graph / 実GPU / 実modelの結果を引き続き分離して保存する。
2. flat Gather修正後の実backbone exportをActionsで生成・照合し、1024長・実encoder feature数profileをSentisでimport / 保存・再読込する。このAPIで独立Pythonの最終768次元へ照合し、CPU / GPUCompute × FP32 / Float16を別判定する。
3. 公式processorのprompt / 特殊token・maskを生成するAPI、ImageEmbedder / AudioEmbedderとGPU前処理をTDDで接続する。text128長の既存APIは維持する。
4. 固定candidateの空Git consumerへRuntime assemblyを導入し、sample・Player・全必須実機へ進む。正式Release / tagは作成しない。
## 実1024長backboneのSentis受け入れ追記

2026-10-13に[CI 38325796546](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38325796546)の修正後graphを監査し、Unity 6000.3.21f1 / Sentis 2.6.1 / Windows / RTX 4070 Ti SUPER / Direct3D12でCPU / GPUCompute × FP32 / Float16の全4条件を実行した。compile成功、4 passed / failed 0 / skipped 0 / inconclusive 0。[固定sourceと全条件](results/multimodal-backbone-sentis-windows-20261013.json)。

実行sourceはproducerのPR merge ref `e521e39af8962bbe9b9a9821e5608e6ade351639`、固定model revisionは`914f7f89142e33e77833254d9c9b90c3cef7303b`。元の実1024長graphのimport失敗を修正後の合格で消さず、[小さい配置TDDと初回失敗](results/multimodal-placement-windows-20261013.json)に保持する。旧stageを別directoryへ移してから新しいhardlinkを配置し、旧sourceの内容を上書きしていない。

保存 / 再読込後、各条件で順序の異なる2入力を参照照合し、有限768次元・単位長を確認した。minimum cosineはFP32 CPU `0.9999999998910367` / GPU `0.9999999999368913`、Float16 CPU `0.9999999998834702` / GPU `0.9999999999423765`。GPU条件の実worker / 出力 / image・audio入力backendはGPUCompute、CPU計算 / feature入力fallbackとGPU readback flushは0件。CPU tokenize / token入力と最終vector取得は境界として扱う。

この4条件は実text重みと合成project済みimage / audio featureのbackbone単独の受け入れ。Runtime sourceをconsumer Editor test assemblyへ配置した結果であり、新Runtime assemblyの固定Git導入・実encoder接続・GPU前処理・最終sample / Player / 他OSの合格ではない。次は開発source配置を外し、固定Git commitのRuntime assemblyを検証する。正式Release / 公開tagは作成しない。

## 固定Git Runtime assemblyの更新consumer受け入れ

固定Git commit `21e598c5b53d5ad73557ee6d606b12770254d73b`を既存consumerへ導入し、Runtime / Profile / 単体テストの開発source overlayをAssetsから退避した。Package Managerのrequested / resolved SHAとGit source、Sentis 2.6.1 / Newtonsoft 3.2.2を読み戻した。型のassembly名は両方とも `EmbeddingGemma.Runtime`。package source監査は98 tracked filesで、改行を正規化したGit blob97件一致と、UPM生成 `_fingerprint`だけを除いたpackage.jsonのsemantic一致。raw bytes全件一致という監査ではない。

更新直後は旧packageが残っており、compileで新型を見つけられずCS0246となった。Package ManagerのResolveを1回だけ要求した後、uloopは受理後のEOF / `UNITY_DISCONNECTED_AFTER_ACCEPT`を返した。`SafeToRetry=false`を守り再要求せず、同じoperationのlock / cache更新とEditor Readyを確認した。これらは導入遷移の失敗・観測切断であり、アルゴリズムのTDD redには数えない。

解決後のcompileとpackage assembly契約は62 passed / fail 0 / skip 0 / inconclusive 0（2026-10-13 00:52:41 UTC）。続いてcompileを省略して実1024長4条件を実行し、同じoperationの報告を追跡した。2026-10-13 01:02:13 UTCに4 passed / fail 0 / skip 0 / inconclusive 0、CLI exit 0を確認。[固定Git更新のreceipt](results/multimodal-git-runtime-upgrade-windows-20261013.json)。CPU / 実GPU × FP32 / Float16で保存 / 再読込、並びの異なる2入力、有限768次元・単位長を照合し、全GPU条件のCPU計算 / feature入力fallback・readback flushは0件。

実text重みと合成project済み特徴量のbackbone受け入れは、開発source内だけでなく配布Runtime assemblyでも合格した。公式Test Framework 1.6.0をembeddedにした既存consumerの更新条件を保持し、最終空consumerの合格とはしない。実画像・実音声encoder、公式prompt / mask構築、GPU前処理、最終API / sample・Player / 他OSは引き続き全体計画の必須gate。共通backbone Runtimeという本PRの範囲と、その後のend-to-end実装を分け、正式Release / 公開tagは作成しない。
