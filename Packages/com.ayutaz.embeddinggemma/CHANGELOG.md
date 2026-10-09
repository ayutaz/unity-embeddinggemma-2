# Changelog

## 0.1.0-pre.1 — Unreleased

- 検証済みテキストRuntimeをUPMへ移行。公開API・Runtime assembly名・既存ソースとGUIDを維持。
- モデル不要の29件のEditor契約テストを配布パッケージへ移行。
- Sentis 2.6.1 / Newtonsoft JSON 3.2.2の直接依存を明示。
- モデル、計測fixture、開発用自動操作ツールは配布対象に含めない。
- Windows Unity 6000.3.16f1の別consumerでローカルフォルダ依存の解決・compile・契約29件成功。元の検証プロジェクトで実モデルCPU / GPUCompute回帰も成功。
- 2026-10-10にPR #6をmainへ統合済み。公開tag、Git URLのEditor導入、macOS / iOS / Androidの受け入れは未完了。
- PR #10で固定文書のcosine検索、同点ID順、所有リソースを解放する検索sessionをmainへ統合。PR #11でText Searchサンプル、明示的なモデル準備・hash監査・配置・更新手順を追加（サンプルはmainへ未統合）。
- Windowsの保存済みfp32 / Float16重み × CPU / GPUComputeの4条件で、固定6文書 / 4queryの全順位がPython参照と一致。元プロジェクトの画面操作も確認。新規consumerのCLI完了応答と画面操作は未完了。
