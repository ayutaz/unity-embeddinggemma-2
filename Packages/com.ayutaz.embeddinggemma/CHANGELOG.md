# Changelog

## 0.1.0-pre.1 — Unreleased

- 検証済みテキストRuntimeをUPMへ移行。公開API・Runtime assembly名・既存ソースとGUIDを維持。
- モデル不要の29件のEditor契約テストを配布パッケージへ移行。
- Sentis 2.6.1 / Newtonsoft JSON 3.2.2の直接依存を明示。
- モデル、計測fixture、開発用自動操作ツールは配布対象に含めない。
- Windows Unity 6000.3.16f1の別consumerでローカルフォルダ依存の解決・compile・契約29件成功。元の検証プロジェクトで実モデルCPU / GPUCompute回帰も成功。
- 2026-10-10にPR #6をmainへ統合。初期UPM検証はローカルフォルダ依存で、Git URLの後続実測は以下に記録。公開tagとmacOS / iOS / Androidの受け入れは未完了。
- PR #10で固定文書のcosine検索、同点ID順、所有リソースを解放する検索sessionをmainへ統合。PR #11でText Searchサンプル、明示的なモデル準備・hash監査・配置・更新手順と、検証済みキャッシュの再利用を追加。
- Windowsの元プロジェクトと新規consumerでfp32 / Float16重み × CPU / GPUComputeの固定全順位・ベクトル一致、CLI 4 passed / skip 0と日英画面操作を確認。
- PR #20で日英表示に必要なfont生成・レイアウトと操作確認を追加。font警告とallocation Logの由来は追跡中。
- PR #24で選択した重みとtokenizerだけを保持する所有cache、ローカルfile / jar URL配置、完全hash、worker使用中lease、キャンセル時の転送解放を追加。成功後1組 / 更新中最大2組のcacheを保持。実Windows file URL / GPU / warm cacheを確認した。実APK / 実機とsample Player画面は未検証。
- PR #25の記録では、PR #24統合SHA `01f3d8682a97053bf4583e14fd8fbed14e2c8cae`を新しい空のWindows Git consumerで導入。UPM 38 / sample EditMode 64 / PlayMode 2、実モデル4条件・全順位 / score、日英入力 / 空入力 / 解放 / warm cache / Play停止を確認。PlayMode CLI接続失敗は同じ実行のXMLで2 passedを回収した。
- Windows検証専用PlayerはPR #23でIL2CPP / compiler Release / High strippingの実buildと実モデル4条件・独立2起動を確認。検証assemblyのpreserve-all条件であり、任意consumerのstripping成功や配布sample Player画面の成功へ一般化しない。
- versionは `0.1.0-pre.1`のまま。macOS Editor / iOS / Androidの必須実測、残る安定性、候補 / tag固定導入、正式Releaseは未完了。
