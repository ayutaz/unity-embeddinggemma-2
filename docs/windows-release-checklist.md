# Windows 0.1.0公開候補の最終確認

これは未公開の候補を完成させるためのチェックリストです。package versionは`0.1.0`へ準備しましたが、
正式tag・Release・新tag consumerの成功を示しません。2026-10-20のGitHub再確認では正式tagはなく、
公開済み4 Releaseはいずれもモデル / 回帰データ / 参照のpreviewでした。

[ローカル準備の確認記録](results/windows-v010-local-preparation-20261020.json)では、package関連34テスト、
版と検証projectの整合、両sample全86ファイルのコピー一致、Runtime tree不変を確認しました。
生成consumerはローカルfile依存であり、公開Git tag・Unity実行の確認には数えません。

## 公開前

旧providerコピーの除去は2026-10-20のユーザー指示で後回しとし、今回の公開待ち条件から外しました。除去完了とは扱いません。現在の送信履歴・新しいpayloadのprivacy gateは下記のとおり維持します。

- [ ] 一つのtask PRに実装・元source付き記録・文書をまとめ、Required CIと新しいWindows音声 / 画像CPU CIの実結果を確認する。skip・timeout・別sourceを成功にしない。
- [ ] `project-privacy`、versioned Git hooks、全送信祖先、PR本文・Release本文・添付物の実bytes、公開identityを確認する。変更画像があれば同じbytesを目視 / OCRで確認する。
- [ ] 必須結果を満たしたPRを明示的noreply authorでマージし、実mainのtree・CI source・署名・identityを読み戻す。mainへ直接pushしない。

## 候補の固定

- [ ] merge後の全SHA、package version、Runtime tree、依存、fixture、モデルdescriptor / 全object hashを記録する。現在のRuntime treeは`7d6e4314552b221d06db881405e7ccff3f449a77`で、版・文書準備によるRuntime変更はない。
- [ ] 対応環境は[実測Windows構成](windows-supported-configuration.md)に限定する。RAM64 GB / VRAM16 GBクラスを真の物理下限と表示しない。他OS・他GPUの合格を追加しない。
- [ ] [コード / モデルのライセンス](model-license.md)と固定model revision `914f7f89142e33e77833254d9c9b90c3cef7303b`を確認する。UPMのApache-2.0と、別取得モデルのLICENSE・NOTICE・MODIFICATIONSを分ける。
- [ ] `v0.1.0`が未使用であることを直前にも確認し、検証済みmainのSHAへ固定する。既存tagを移動しない。tag公開前にもprivacy gateを満たす。

## 空consumerからの実検証

次のURLは予定する正式tagです。現時点で存在する導入pinとして案内しません。

```text
https://github.com/ayutaz/unity-embeddinggemma-2.git?path=/Packages/com.ayutaz.embeddinggemma#v0.1.0
```

- [ ] 空consumerで上記tagを解決し、`manifest.json`のURL、`packages-lock.json`のGit source・tag URL・resolved SHA、実PackageCacheのpackage version・配布全bytes・依存を固定SHAと照合する。既存のSHA専用consumer監査をtag監査の代わりにしない。
- [ ] Unity6000.3.21f1 / Sentis2.6.1でcompileし、Text SearchとImage and Audio Searchを取り込む。依存解決だけで実行成功としない。
- [ ] 公開元からFP32 / Float16重みの全6 moduleを明示取得し、descriptor・全objectのサイズ / hashと法的表示を確認する。取得と実モデル推論の成功は別々に記録する。
- [ ] 実Windows Editorで両形式のtext全role、全画像 / 音声profile・role、Texture / RenderTexture・PCM、混在索引・横断検索の順位 / scoreを元参照・元許容差と照合する。
- [ ] 同じ固定consumerからWindows IL2CPP / Release / High Playerをbuildし、両形式で全入力・検索・解放・正常終了を確認する。Editor成功をPlayer成功として数えない。
- [ ] actual GPU backend、中間readback・計算CPU fallbackがないこと、load / cancel / retry / 切替 / 解放を確認し、終了code・process終了・元receiptを保持する。既存sourceの結果を新tagでの実行に読み替えない。

## 正式Release

- [ ] 全必須結果を確認してから候補表記を確定し、CHANGELOG・Release notes・対応構成・導入手順と実tagを一致させる。検証後に実行codeを変えた場合は該当検証をやり直す。
- [ ] `v0.1.0`の正式Releaseを公開する。モデル、内部検証Player、個人の録音、原本log、大きい検証artifactをSDK添付物にしない。新たな添付物は実bytesを個別監査する。
- [ ] GitHub Releaseの正式 / draft / prerelease状態、tagのresolved SHA、配布package version・ライセンス・公開手順を読み戻して一致を確認する。全項目の証拠が揃った時点で正式公開完了と判定する。

ローカルのpackage監査・空consumerファイル生成は準備検査です。Unity起動・Git取得・公開tagの実在・
GPU動作・正式公開の証明にはしません。工程別の進捗は[現在の実行計画](next-release-plan.md)へ記録します。
