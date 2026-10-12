# Unity 6000.3.21f1更新と終了比較

2026-10-12にユーザーが6000.3.21f1への固定版更新を承認した。公式revisionは`c02631ffc030`。関連する[公式修正UUM-146676](https://unity.com/releases/editor/whats-new/6000.3.21f1)はWindows accessibilityの終了クラッシュであり、今回の問題との一致は実行比較で判断する。

## 更新範囲

- 作業ブランチのProjectVersion、空consumer生成、手動Unity CPU CIの実行版を21f1にした。
- 新しいPlayer build provenanceは21f1を必須にする。過去16f1の保存結果は書き換えない。
- Sentisは2.6.1のまま。既存の監査済みモデル、Python参照、Library / build cacheを再利用する。
- 配布packageのminimum 16f1は以前のEditor検証に対応した値で、21f1のnative検証が済んだ段階で配布条件を見直す。現在の21f1指定を実モデル検証済みとは扱わない。
- Androidの過去APK / bundle監査は16f1 provenanceのまま保持する。21f1のAndroid実機合格へ読み替えない。

## TDDと導入

版指定Python契約は2 failedを確認してから実装し、関連30件がpassed。Player provenanceの新21f1受け入れ・旧16f1拒否は2 failed → 2 passedを確認した。この2件は旧Editor上のmetadata契約で、実21f1での起動や精度の証明ではない。

公式Windows Editor / IL2CPP installerを取得し、Authenticode Valid / Unity Technologies SF署名を確認した。Editorはworkspace内の専用ディレクトリに導入し、実ファイルのProductVersion `6000.3.21f1_c02631ffc030`を確認。IL2CPP moduleとnative回帰の完了は下のgateで別途確認する。installer・Editor本体・生ログはGitに入れない。

## 実行順序とgate

1. 保存sceneがclean・Play / compile / Test Runner停止を確認し、旧Editorを正常終了する。終了後のPID消失を確認してから21f1を1つだけ起動する。
2. 同じconsumerをuloopでcompile・契約確認する。実行中のoperation / PID / 保存結果を継続観測し、観測timeoutだけで再実行しない。
3. model-freeのnative終了を比較する。旧通常モードの終了アクセス違反と、同じ実行経路での21f1終了を分けて記録する。モデル欠落の想定exit 1を実モデル合格と数えない。
4. sample IL2CPP / Release / High strippingで日本語・英語GUI、全4query / 各6順位・score、空入力・欠落モデル・解放を固定Python参照と照合する。独立2起動がexit 0で、初回 / warm cacheの監査も通ることを必要条件にする。
5. 新固定Git candidate導入と実モデルCPU / GPU × FP32 / Float16の回帰を必要範囲で確認する。Android / Mac / iOS実測と正式Releaseの条件も維持する。

21f1のnative終了・実モデル・独立2起動・他環境はこの時点では未完了。PR #31をdraftのまま進める。
