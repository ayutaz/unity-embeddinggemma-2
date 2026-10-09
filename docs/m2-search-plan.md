# テキスト検索サンプルとモデル準備の実装計画

開始日: 2026-10-10。対象は利用者向け一覧の1（検索サンプル）と2（モデル取得・変換・配置手順）。[M2計画](m2-plan.md)のA / B / Cを進める。macOS・モバイル・公開リリースは後続であり、この作業でM2全体の完了とはしない。

| 段階 | 実装 / 成果物 | 合格の証拠 |
| --- | --- | --- |
| A | 固定の日本語 / 英語文書・query、cosine順位計算と同点時ID順、検索のPython参照生成 | Python / C#のTDD red → green。Actionsで固定モデルrevision・prompt・tokenizer・128長の参照ベクトルと全順位を生成。出力のsource SHA・SHA-256を監査 |
| B | UPMのTextSearchサンプル、事前埋め込み・query入力・順位表示・CPU / GPU選択・モデル未準備 / 不正入力 / 実行失敗表示・終了時Dispose | 所有リソースとエラーの契約テスト。Windows Editorでfp32 / Float16重み × CPU / GPUComputeの全query順位をPython参照と照合、skip 0。UIの準備 → 入力 → 表示 → 終了を操作し画像を残す |
| C | Actions成果物の取得、監査、Sentis変換・配置・更新、サンプルの導入手順 | 重みを含まない新規checkout / 空のUnity consumerへサンプルを導入し、CI参照を配置して起動・検索を再現。revision・生成commit・hash、必要操作、失効時再生成を文書化 |

各挙動は先にテストを書き、意図した失敗を確認してから実装する。重いモデル参照生成・変換と全Python検証はActionsへ寄せ、ローカルでは新規契約の小さい確認と、クラウドで代替できない必要最小限のUnity CPU / GPU操作を行う。Pythonはtools/とuvのみ。

既存TextEmbedderとTextModelFileを使用し、同期・メインスレッド・batch 1 / length 128 / 768次元を維持する。最終ベクトルだけCPUへ戻して小さい文書群を全件比較する。GPU非対応時にCPUへ自動で切り替えず、利用者がbackendを選択する。

モデルは自動ダウンロードしない。利用者が明示的に用意した`.sentis`とtokenizerを読み、各検索で再保存・再量子化しない。重み・大きい生成結果はGit / LFSへ含めず、小さい実行要約と再現手順を記録する。

実装ブランチは`feat/m2-text-search`。計画更新PR #9を基礎にし、実装はPRへ提出する。mainへ直接pushせず、マージは依頼があるまで行わない。未実行・失敗・GPU skipは合格と分け、全段階の証拠が揃うまでこの作業を完了扱いにしない。
