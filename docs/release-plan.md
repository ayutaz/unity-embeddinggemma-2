# 正式リリースまでの計画

確認日: 2026-10-18。詳細は[ロードマップ](roadmap.md)、対象範囲は[ゴール](goal.md)に従います。

## 現在地

- 正式Releaseは未作成。PR #86・#87・#89・#90はすべてDraft・未マージです。
- 音声の別世代候補は[全6信号の数値CI](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39106618371)が成功。2 profile・824,320値で元の許容差超過は0、取得した実データの独立再監査も一致しました。
- この結果は候補の実GPU・consumer・Playerの合格を意味しません。旧世代のnative PT2モデル2件のbyte同一復旧と、必須CIの復旧は未達です。
- #90の固定Git版 `6350aa6` を新規consumerへ導入し、Libraryを再利用せず、Windows IL2CPP / Release / High strippingの実buildに成功しました。最小consumerのコンパイルで見つかったUnityWebRequest依存の不足も修正済みです。
- 同Player・既存FP32 bundle・Direct3D12で、音声2 profileは成功・GPU確認あり・process exit0です。通常18件・warm18件を元のhash固定参照から独立再計算し、最小cosine `0.999999709769265`、全Worker監査・run identity・正常終了を確認しました。8件の意図的な破損controlも拒否しました。画像squareの先行試行は推論前のGPU監査で失敗し、後続2 profileは未実行です。Float16、最終モデル候補との対応付けは残っています。Direct3D11での先行音声試行の失敗も保持します。
- 画像失敗の原因は、入力をそのまま出力へ渡す17区間を空graphとして一律拒否した監査でした。全181 Workerのmetadata調査と実保存モデルの入出力を確認し、CPU計算・中間flushは0でした。出力が入力を参照すること、定数なし、GPU所有、CPU割り当てなしを検証する修正を #90 の `89b1ab9` に反映しました。8 Redの後、関連30 Green・skip0・Editor exit0です。修正版の実Player検証は別gateとして残します。[checkpoint](results/media-player-input-alias-audit-20261018.json)
- その後、固定Git `89b1ab9` の新規consumerで実build・exit0とコンパイル対象62ファイルのhashを確認しました。[修正版Playerの実行](results/media-player-retained-bundles-gpu-89b1ab9-20261018.json)は、既存FP32 / Float16重みbundleの画像3・音声2、全10条件でGPU監査・キャンセル回復・解放・exit0を確認しました。通常72件・warm90件の独立再監査と40件の破損control拒否も成功しました。Float16変換記録と553モデルの対応も両descriptorへ照合済みです。新世代モデルの採用、text・横断検索、実操作、独立warm起動、資源計測、他端末、必須CIは未完了です。
- #89の過去head `3949d1a` の[CI 39112717115](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39112717115)は、Python全4環境・package・lint・通常referenceが成功、固定元モデルの6信号・集計・Required CIは失敗しました。#90の過去head `6350aa6` の[CI 39112580190](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39112580190)も不合格です。#89の現head `6052fe4` の[CI 39117532718](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39117532718)は終了・不合格でした。Python全4環境・package・lint・通常referenceは成功しましたが、音声6信号と集計は固定参照の取得段階で失敗し、Required CIも失敗しています。候補の数値成功とは別に復旧が必要です。
- 以前のsource・既存bundleでのGPU動作実績は保持し、最終候補に対応する確認と区別します。

### 音声候補の保存checkpoint

[保存CI 39109531607](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39109531607)が成功しました。デバッグ情報だけを除いた配布用コピーについて、両profile・全6信号のnative出力が除去前後でbit一致しています。配布artifact `12129783259` は `2027-01-16T07:56:21Z` まで保存されます。[全体読み戻し](results/audio-candidate-distribution-full-readback-20261018.json)で、約2.45 GBのZIP全体・両モデルのSHA256、全CRCと私的な2コピーを確認しました。両配布モデルのhashは元の候補からのローカル導出とも一致しました。

候補producerの元graph2件・raw入力6件・JSON2件の計10ファイルは、元のhashで私的な2コピーを確認しました。消失した旧世代PT2の復旧とは区別します。保存実行sourceは `6bfc223` です。必須CI・新世代候補の実GPU・consumer・Playerの合格には読み替えません。

[元の数値参照とraw入力の保存CI](results/original-audio-input-preservation-readback-20261018.json)も成功しました。artifact `12130281114` は90日保存され、ZIP全体のSHA256・CRC、元JSON2件とraw入力6件のhash、私的な2コピーを確認済みです。取得先を固定した `0d13a3d` の[実取得](results/original-audio-preserved-acquisition-0d13a3d-20261018.json)も8ファイルすべて一致しました。モデルの実行・生成は伴っていません。

旧native ONNXは両profileとも元auditのhashで保持されており、別の私的コピーも検証しました。[候補の実bytesとの比較](results/audio-candidate-native-onnx-identity-readback-20261018.json)は両profileで演算同一性が不一致でした。候補は過去のdivision-SiLU版と同じhashであり、差をmetadataだけと扱いません。[保持GPU変換モデルとのinitializer比較](results/audio-candidate-gpu-preparation-initializers-20261018.json)はfull profileの型・値multiset一致までで、名前付きの対応や計算経路全体の同一性は未証明です。次は変換手順と重みの対応を固定し、新世代候補として実GPU検証します。

### 候補から直接生成したGPU変換の起点

保存候補のgraphとTensorの対応を直接引き継ぐ経路を追加しました。2か所のLayerNormだけを既存Welford8演算へ置き換え、コピーと内部Softmaxは同じ値・型を保つexport表現へ橋渡しします。最初の2回の実CI失敗と、修正前の失敗テストを保存しています。関連49件とworkflow回帰18件・actionlintが成功しました。

[実CI 39117103059](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39117103059)は、source `0cee4c7` の両profileで生成・metadata監査・uploadが成功しました。[全体読み戻し](results/audio-candidate-gpu-seed-readback-0cee4c7-20261018.json)では、実ZIP全体・モデルのSHA256、全CRCと私的な2コピーが一致しています。各profileの元Tensor 1,067件の対応・変換後hash保持と、新しい学習済み重みを追加しないことを記録しました。これは変換起点の生成で、数値・GPU・新世代採用の合格ではありません。

## 直近の優先作業

1. hashを固定し保存できた両profileの変換起点へ、GPU用の演算精度調整・分割を適用します。元Tensor・係数の保持を各段階で確認し、元のencoder要素別許容差・全6信号・全必要roleで実GPU検証します。既存bundleの全10条件成功や起点の生成だけを、新世代の採用判定には使いません。
2. 同じ最終候補へtext・横断検索・sample操作と計測を対応付け、Playerの残件を閉じます。独立warm起動、実マイク、process RAM / VRAMと実GPU時間は、今回の内部warm反復・managed heap・wall時間から推定しません。
3. 取得pinと必須CIを復旧し、PRを整理します。CIが失敗したままの統合や、旧モデルの同一bytesを復旧した扱いは行いません。
4. 同時にMac・iOS・Androidの実行環境を確保します。端末待ちをWindows完了後に発見しないよう、必要環境と担当を工程5の前に確定します。

Playerの生ログと元の実行証拠は非公開で保持します。重い再実行は具体的な修正の検証として行い、source・モデル・終了条件を固定します。

## 実行順と完了条件

工程1を進行中です。工程2〜4には既存bundleでの部分的な成功がありますが、最終候補での完了判定は残っています。工程5の実機確認、工程6の最終配布監査、工程7の正式公開も未完了です。まず工程1の成果物として、両profileの再現可能なGPU変換・分割、全6信号の元参照照合、取得pinと必須CIの復旧をそろえます。変換・静的な形状監査・単体テストの成功だけで工程1を完了にしません。

| 順序 | 実施内容 | 完了の判断 |
| --- | --- | --- |
| 1 | 音声候補と参照データの保存・取得経路を確保し、候補の実GPU・consumer・Player検証を未統合branchで実施。配布モデル内の絶対パスを含むデバッグ情報も監査 | 原本の参照値・許容差を維持して必要な検証が成功。配布用にデバッグ情報を除く場合は計算と重みの不変を検証し、変更後のhashを別途固定。必須CIを条件変更で迂回しない |
| 2 | 新規利用者向けのモデル準備を確立。空checkout・空cacheからtext・画像3・音声2の全6 moduleを準備 | 公開手順だけで準備でき、source・model hashが一致。欠落・破損・中断・期限切れから復帰できる |
| 3 | 日英検索、画像プレビュー、音声再生・途中停止、実マイク、キャンセル・再試行・解放を最終候補で確認。性能の不足を修正 | 権限拒否・deviceなし・非対応入力・中断復帰を含め成功。初回/warm、CPU/GPU時間、RAM/VRAMを測定範囲とともに記録 |
| 4 | WindowsのIL2CPP / Release / High stripping Playerで3入力・全必要role・横断検索を確認 | 元参照のvector・順位・score、実GPU使用、fallback/readback監査、独立cold/warm起動、解放・正常終了を確認 |
| 5 | macOS EditorとiOS / Android実機で受け入れ | 3入力の実GPU結果、権限、suspend/復帰、メモリを端末・OS・graphics APIごとに確認。未実行・skip・CPU fallbackをGPU合格に数えない |
| 6 | 最終source・モデル・依存を固定し、配布物・README・API・sample・CHANGELOG・利用条件・最低要件を整備 | 最終候補で全必須回帰が成功。履歴・公開payload・画像の目視/OCR/hashを監査。旧PR/cacheの個人情報対応も追跡 |
| 7 | version・tag・GitHub Releaseをそろえて公開し、公開tagから導入し直す | 公開物の読み戻し、新規consumer導入、sample smokeが成功 |

工程1と工程2〜4の検証は重なるため、未統合候補で先に検証を進めます。新モデルの採用条件を満たした後に取得pinと必須CIを復旧し、最新headのCI・差分・privacy確認を経て #89 → #86 → #87を統合します。#90はbaseである#89の統合後に差分を再確認し、Player検証の根拠とともに統合します。マージの順序と検証の開始順を混同しません。

## 並行準備と日程

- macOS・iOS・Androidの実機と署名・build環境を先に確保します。現在記録されているAndroid接続端末とself-hosted runnerは0、Mac・iOS環境は未確定です。実行開始時に再確認します。
- 重い取得・変換・build・回帰はGitHub Actionsを優先し、ローカルUnityの重複起動を避けます。同じモデル探索や生成を根拠なく繰り返しません。
- 各checkpointでは「合格・失敗・未実行」「実行source/model」「次の残件」を記録します。テスト数だけでリリース進捗を表しません。
- 次の進捗報告は、変換CIの終了、元参照に対する実GPU照合の終了、必須CIの復旧を区切りにします。失敗時は原因と次の修正を報告し、同じ未解決条件で重い再実行を繰り返しません。
- リリース日は未確定です。音声候補のGPU採用条件と必須端末の実行見通しが立った時点で、実測時間に基づき日程を更新します。端末未確保のまま完了日を約束しません。
- 公開表示名 `yousan` は維持します。元画像や元の数値証拠を加工・再生成して成功の証拠に置き換えません。
