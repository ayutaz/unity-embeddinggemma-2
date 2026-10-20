# 正式リリースまでの実行計画

確認日: 2026-10-20（JST）。[ゴール](goal.md)のテキスト・画像・音声、横断検索、Sentis GPU、Windows / macOS / iOS / Androidの必須条件を維持します。過去の検証記録は[検証状況](status.md)、従来の計画は[release-plan](release-plan.md)に残します。

## 現在地

公開mainは `e72b5cb`（[#112](https://github.com/ayutaz/unity-embeddinggemma-2/pull/112)まで統合済み）。採用head `874c1c8` の全必須CIと実13数値archiveを独立照合し、実mainの同tree・署名・公開identity・全祖先も確認しました。[統合記録](results/platform-ci-integration-874c1c8-20261020.json)を基準に、後続sourceの結果は別に判定します。

| 項目 | 確認した結果 | 残る条件 |
| --- | --- | --- |
| WindowsのSDK GPU動作 | 固定sourceの実モデル検証結果あり。現mainのSDK実行コードはその検証sourceから不変 | 全必須環境、最終candidateの導入・性能・運用検証。package内文書は変更済みで、package全bytesの一致は主張しない |
| #112のPythonテスト | Windows / Ubuntu × Python 3.13 / 3.14の4環境で各2,612件実行合格、失敗・error 0。実XMLを独立照合。修正対象CLI 2件も各環境で実行合格 | 各環境のGPU関連skip 9件はGPU合格に数えない |
| #112の必須CI | [全15 job成功](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39358511727)。実13数値archive、元824,320値、CPU encoder 120 capture / 30,822,400値を独立照合し許容差超過0 | このsourceの数値検証。全体性能・実GPU・正式公開の合格とは分ける |
| build前の固定入力配置 | 後続変更で既存native build CLIへ接続。関連128件合格・skip0、元28ファイル34,474,958 bytesの実配置と全hash・私的2コピーを照合 | 本変更の必須CI、原モデル取得、端末persistent data配置、実build・実機GPUは未完了 |
| macOS / Android / iOS CI | [全3 preflightがfailure](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/39358511760)。理由は `unity_license_missing`。Editor導入・Unity build・Xcode compile・実モデル実行は未開始 | 対応するUnity CIライセンス経路の確立。repo Secrets 0、self-hosted runner 0を確認 |
| hosted Mac GPU | Swift compileとMetalの16整数照合は成功、不一致0。実3 archiveを独立照合 | 小さい能力probeの結果。実Sentisモデル・必要メモリ・性能は未合格 |
| 正式公開 | 公開済み4件はいずれもpreview | 正式SDK tag / Releaseと、そのtagを使う新規consumerは未完了 |

CPUのPython参照・比較は検証用です。製品の画像 / PCM前処理、encoder、backboneはGPUで接続し、中間CPU readback・計算fallbackを禁止する要件を維持します。

## 優先順と完了条件

| 順序 | 実施内容 | 完了条件・次工程への移行 |
| --- | --- | --- |
| 1 基盤の統合・整理 | #112は全15 job・実13数値archive・privacyを確認して統合済み。固定入力配置と計画更新を後続の1 PRへまとめ、実mainをbaseに検証する | #112の実main tree・署名・公開identity・全祖先は確認済み。後続変更の必須CIは別途必要。別workflowのライセンスfailureを未解決として保持 |
| 2 CIで各platformをbuild | Unityライセンスの対応経路を1つ確定し、安全に設定。既存入口でmacOS Editorテスト、Android ARM64 / IL2CPP / High / Release APK、iOS exportと署名なしXcode compileを並列実行。原モデル / fixtureのhash検証付き取得を接続 | 各platformで実Editor・build・正常終了・成果物hashを確認。iOS export / compile / 署名 / 実機起動は別判定。buildだけでモデル実行を合格にしない |
| 3 実GPUと実機を検証 | macOS Metal、iOS / Android実機で全入力・全role・混在検索、元数値gate、actual backend、fallback / readbackを監査。権限拒否、キャンセル、中断 / 復帰、切替、解放・正常終了も確認 | Windowsを含む全必須環境の結果表に成功・失敗・未実行を記録。実機・実モデルの証拠が揃って通過。tiny probe、エミュレータ、skipは代用しない |
| 4 性能と公開前整理 | 同一hardware・source・モデル・入力・仕事量でCPU/GPU、初回 / warm、各stage、RAM / VRAM、解放を測定。画像CPUの未合格条件を元許容差で追跡。最終公開物・履歴・実payloadを監査し、旧providerコピーの除去も確認 | 数値gateに通る比較表と実測に基づく最低要件。既知の失敗を保持し、現行履歴とprovider側残存の確認を分けて完了 |
| 5 正式SDK公開 | 版・CHANGELOG・license・取得手順を確定し、固定tagを空consumerへ導入。モデル取得→全入力→検索→解放・終了を通して正式Releaseを公開 | 工程2〜4の必須条件を満たし、公開source / model hashと新規導入結果が一致。previewを正式SDK Releaseに数えない |

工程1の基盤統合は完了しました。後続変更の検証と並行して工程2のライセンス経路、工程3の端末・署名、工程4のGPU資源・provider対応を準備します。工程2〜3は準備できたplatformから進め、未準備platformを理由に全体を止めません。ライセンス未設定のまま同じpreflightを再実行しても前進しないため、環境変更後に再開します。

## 環境依存と担当

| 条件 | 次の具体的な対応 | 担当・再開条件 |
| --- | --- | --- |
| Unity CIライセンス | 利用可能なライセンス種別とrunnerに対応する認証経路を確定。値をchat・Git・公開ログへ出さず設定する。ローカルPersonalのentitlement XMLをCIへ流用しない | 開発側が手順を準備。アカウント管理者による必要設定後、実preflightがreadyになれば工程2へ |
| Mac / iOS / Android | 利用可能なhost・端末・接続、iOS署名を確定。hosted Macの小さいprobeだけでモデル容量が足りると仮定せず、モデル配置前に容量を判定 | 開発側がbuild / 検証を実装。端末・必要署名が利用可能になれば工程3へ。ローカルAndroid moduleのUAC待ちはhosted APK buildと分離 |
| GPU性能測定 | 開始前に空きVRAMを再確認。音声測定の既存開始条件は14 GiBで、SDK最低要件ではない。必要な資源が足りない場合は利用可能な既存環境へ切替 | 資源確保後に工程4の実測へ。他アプリ終了・有料環境利用は既存の許可範囲を確認 |
| 旧providerコピー | 既存Support依頼の進展と匿名取得の可否を、providerの返答・変更時や最終公開判定前に確認 | provider対応後に除去を再検証。問い合わせの追加送信や同じ状態の頻繁な再照会を作業成果に数えない |

現在の正式公開日は未確定です。工程1の基盤統合後の残件は表のとおりです。対応ライセンス・必須端末を確保した時点と最初の実build完了時に所要時間を見積り直します。CIの経過時間を端末確保・署名・provider対応の日数の見積りへ流用しません。

## 長期化を防ぐ進め方

- 重い処理はGitHub Actionsを優先。成功済みの同source・同モデルの変換や検証を無条件に繰り返さず、変更と結果の適用範囲で再実行を判断します。
- #112は検証対象headを維持して統合しました。計画だけで重いCIやPRを増やさず、固定入力配置の実装と本計画を後続の1 PRへまとめます。
- 実装はTDDで変更をまとめ、task branch / PRで進めます。mainへの直接push、失敗に合わせた元参照・許容差の変更は行いません。
- 各工程の報告は「実行source・実行範囲・成功 / 失敗 / 未実行・次の開始条件」に統一。外部条件待ちは何を待っているかを明示し、同じ調査や監視を無期限に繰り返しません。
- 公開前にはproject-privacy skill、versioned Git hooks、全祖先と実payloadを確認。画像を公開する場合は実bytesの目視とOCRを行います。公開名 `yousan` は維持します。
