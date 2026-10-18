# 現在の対応範囲と検証状況

更新: **2026-10-18**。公開履歴整理後のmainは `4b5bf72`。開発版 `0.1.0-pre.1`、公開tag・正式Releaseはありません。現在は個人情報対策の回帰修正、固定音声参照データの復旧、GitHub側の残存個人情報の除去を優先します。

## 現在の停止条件と次の作業

| 項目 | 確認済みの状態 | 次の完了条件 |
| --- | --- | --- |
| hook | 日本語の正当な編集がWindowsのCP932入力処理で拒否されることを再現し、JSONをUTF-8のバイト列から読む修正で14テストが成功。実行中のhookにも同じ修正を適用 | 不正な個人パスの拒否を維持し、追加の公開前チェックとともに#89へ統合 |
| CI | 関連84テストとpackage検証CIは成功。全Python CIは4環境とも失敗。Linux 3.13では12 fail / 1,541 pass | 元の失敗時保持条件とprivacy guard成功条件の両方を検証するようworkflowテストを修正し、Windows/Linux × Python 3.13/3.14を再確認 |
| 固定音声参照 | Actions記録の削除で必須の元モデル成果物も失われた。参照JSONと監査JSONは保持、元モデルグラフ2件は未復旧 | 元SHA256に一致する完全バックアップと取得経路を確保し、必須の元音声参照CIを再実行 |
| GitHub側の個人情報 | writable branchの履歴修正済み。旧PR/cache除去と成果物復旧をSupportへ依頼済み。受付情報は私的記録に保管 | 対応結果を照合し、旧PR refs・キャッシュ・残るActionsメタデータを再監査 |
| PR整理 | #89（privacy）、#86（取得）、#87（性能）はDraft。全体Required CIの成功・全機能完了は主張できない | #89、#86、#87を各受け入れ条件と最新headのCIに基づき順に判定 |

追加の大量削除は停止しています。再開前に固定artifact IDの依存を棚卸しし、必要なモデル・参照バイトの私的バックアップとSHA256を確認します。メタデータ保存だけを成果物の退避とみなしません。復旧不能の場合は新しい参照世代の計画と差分を別に示し、原本と同一の実績には置き換えません。

## これまでの機能・GPU検証記録

以下は各固定sourceの従来の実測です。履歴整理後の最新headで全条件を再実行した証拠や、元成果物を現在も取得できる保証にはなりません。従来の負荷軽減監査のRuntime Git treeは `7d6e4314552b221d06db881405e7ccff3f449a77` でした。

[ランタイム修正の統合監査](results/runtime-ready-integration-2a29c68-20261018.json)では、単独の統合候補 `2a29c68` でpackageとmainの既存2サンプルのEditMode回帰517件（GPU演算等120件を含む）がpass、fail0 / skip0 / exit0でした。Runtimeとpackage依存は、全6 module×FP32 / Float16重みの実GPU参照32 vectorを確認した固定source `d7b2233` と一致します。これはsource一致に基づく既存の実モデル証拠と、新しい小規模GPU・既存sample回帰を組み合わせた監査であり、今回の32 vector再実行や最新UXの操作合格ではありません。

9演算のwrite-only出力で不要なCPU storageの確保・ゼロ初期化・転送を省き、画像segmentとtextの大きな定数を8MiB単位でGPUへ転送し、softmaxの行ごとのdispatchを制限します。元の係数・参照・許容差・GPU経路と演算順序は維持します。公開モデルのempty-cache取得、実マイク、最終candidateの全操作、SDK専有load/release GPU時間・allocation、他環境・正式Releaseは未完了です。これらをランタイム修正の先行統合で完了扱いにはしません。

[導入](../README.md) / [ドキュメント一覧](README.md) / [今後の計画](roadmap.md)

## 統合済みの機能

| 項目 | 現状 | 主な根拠 |
| --- | --- | --- |
| テキスト推論・検索 | UPM、保存・Float16重み、Text Searchを実装。固定Git consumerとWindows Playerの実モデル検索を確認 | [PR #31](https://github.com/ayutaz/unity-embeddinggemma-2/pull/31)、[Player監査](results/m2-current-main-player-acceptance-20261017.json) |
| 画像・音声API | 実Texture / PCMからGPU前処理→encoder→共通backbone→正規化768次元。Query / Document / Rawを固定入力で検証 | [PR #79](https://github.com/ayutaz/unity-embeddinggemma-2/pull/79)、[元API完了監査](results/goal4-current-completion-audit-20261016.json) |
| 画像・音声検索サンプル | Package ManagerからImport可能。実ファイル登録、混在検索、module切替、キャンセル・復帰・解放 | [PR #83](https://github.com/ayutaz/unity-embeddinggemma-2/pull/83)、[サンプル検証](multimodal-search-sample.md) |
| モデルbundle・cache | 6 moduleの準備処理、サイズ・SHA256照合、選択moduleだけのcache・破損修復・排他lease | [サンプル検証](multimodal-search-sample.md)。空checkoutからの利用者向け自動準備は未整備 |

PR #31と#83はマージ済みです。#83の検証head `613ada824bd13a297660d8c4de4c2c14b4e317e8` は[CI 38939320058](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38939320058)の15 jobすべて成功。当時の統合直後mainと同じtreeです。これはコード統合の確認であり、正式リリースの完了ではありません。

## 確認済みの範囲

実測はそれぞれの固定sourceに対応します。異なるsource・Unity版の結果を、最新headの全条件再実行として合算しません。

| 検証 | 結果・範囲 | 固定記録 |
| --- | --- | --- |
| 元画像encoder | 6画像・3 profile、778,240値が参照とbit一致 | [画像結果](results/m3-adjacent-reciprocal-all-image-gpu-passed-20261015.json) |
| 元音声encoder | 6信号・2 profile、824,320値で元許容差超過0 | [音声結果](results/m4-staged-original-encoder-and-final-text-gpu-regression-20261016.json) |
| 元APIの最終vector | 画像・音声それぞれ全6入力×3 role、各18条件合格。画像RenderTexture・復帰も確認 | [画像API](results/m3-final-image-api-all-cases-all-roles-gpu-20261016.json)、[音声API](results/m4-real-pcm-all-roles-staged-gpu-20261016.json)、[RenderTexture](results/m3-rendertexture-recovery-and-text-gpu-regression-20261016.json) |
| 保存bundleから実画像・PCM | 5 profile・12条件、最終Document vectorの最小cosine 0.9999998052055729。5プロセスすべて正常終了 | [全profile](results/multimodal-prepared-media-all-profiles-gpu-20261017.json) |
| 混在サンプル | 実音声・画像、固定4 query、GPUキャンセル→再実行→切替→解放・終了。修正後warmのホストピーク約9.49 GiB | [混在実行](results/multimodal-search-mixed-sample-cache-retirement-20261017.json) |
| 実PlayMode画面 | importしたsceneで登録・検索し、日英表示を目視。手動のマウス・キーボード操作は未検証 | [画面監査](results/multimodal-search-playmode-ui-20261017.json) |
| Unity全体回帰 | package + Text Search + Image and Audio Search、505 pass / fail0 / skip0、正常終了 | [全体回帰](results/multimodal-sample-canonical-regression-20261017.json) |
| Windows text Player | IL2CPP / Release / High stripping、独立cold / warm、4 query×6順位・score、FP32 GPU、正常終了。CPU計算fallback0 | [Player監査](results/m2-current-main-player-acceptance-20261017.json) |

元APIは主にUnity6000.3.16f1、最新サンプル・consumer・Playerは6000.3.21f1です。いずれもSentis2.6.1 / Windows / RTX4070 Ti SUPER / D3D12の固定実測です。Playerの合格範囲はtext検索で、画像・音声Playerの合格を含みません。

## GPUとメモリ

画像・音声の受け入れ経路はGPUComputeで実行します。中間tensorのCPU readback・計算fallbackを許可せず、入力の取得・upload、CPU tokenizer、最終768次元のCPU取得を境界として扱います。

音声キャンセル後の再実行でホストメモリが16,757,751,808 bytesに達した失敗を保持しています。GPU fence完了を待つ前処理とcache解放へ修正後、同じ混在warm手順は10,190,360,576 bytesで正常終了しました。修正前cold合格と最終vectorはbit一致しています。

全profile検証の最大ホストピークは12,090,900,480 bytes（約11.26 GiB）。これらは実測であり、VRAM必要量・全端末の最低メモリ・長期安定性・速度の保証ではありません。Float16重み保存も高速化の保証にはしません。

## 未完了・未検証

- 空のcheckoutから画像・音声bundleを準備する公開手順と配布経路。
- 最終統合candidateを新しいconsumerへ導入し、全モダリティを手動で一通り操作する最終受け入れ。
- 結果画像のプレビュー、結果音声の再生、明示的なマイク収録・停止・権限失敗の処理。
- 同一candidateでの初回 / warm反復、VRAM・ホストメモリ・ロード・Editor負荷の系統的な測定。
- 画像・音声検索のWindows Player、macOS Editor、iOS / Android実機での精度・操作・cache・正常終了。Androidの過去のAPK build成功は実機GPU合格ではありません。
- 全モダリティの採用Float16形式・明示CPU経路の環境別受け入れ、最終candidate / tag固定導入、正式Release。

過去のCPU対ONNX診断失敗、旧同期音声のメモリ上限停止、未実行・skipは成功に置き換えません。[サンプル検証](multimodal-search-sample.md)と[過去の設計・検証資料](README.md)に保持しています。次の順序・依存・公開条件は[ロードマップ](roadmap.md)を参照してください。
