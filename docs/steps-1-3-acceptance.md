# 工程1〜3の作業基準と受け入れ

2026-10-14。ユーザー指定の範囲は、(1)基準・PR依存整理、(2)画像・音声encoder数値不一致修正、(3)実入力→最終768次元APIの受け入れ。sample・正式Releaseは今回の完了条件へ置き換えない。

## 基準とPR依存

確認mainは `c868df5ca67f1ca8a6173a0f8e81f9f0bb7367e1`。CI run 38428339728は8 job成功。mainの既存ローカル変更は維持する。作業branchは `feat/multimodal-numerical-acceptance`。統合はこの作業branch内で行い、mainへmergeしない。

全22件のopen PR・head SHA・base・CIは [固定snapshot](results/steps-1-3-start-checkpoint-20261014.json) を参照する。作業開始時の状態であり、各実行のsourceへ付け替えない。

| 系統 | 依存と役割 | 受け入れ状態 |
| --- | --- | --- |
| M2 | #31→main。#36の導入helperを含む | Windows条件付き実測。最終空consumer・標準環境・他OS未完了 |
| 画像元参照 | #41 / #53→main | Float16 CPU/GPU計6条件合格、FP32計6条件不合格。#53はnative文脈gate未達 |
| 音声Python | #40 / #60→main | native保存参照は合格、ONNXはsilence等の数値不合格 |
| API | #62→main、#64→main、#65→#62（#64を含む） | Audio/Image APIの合成契約合格。実モデルgateは別 |
| CPU画像 | #67→#65、#68→#67 | 前処理60条件合格。実encoderはFloat16合格・FP32不合格。合成最終chainにも1件不合格 |
| GPU定数 | #69→#67、#70 / #71→#69 | 分割転送の実backbone24件合格、音声全chain24件は最終cosineのみ合格。IL2CPPは重みなしprobe |
| 音声観測 | #72→#71、#73 / #74→#72、#76→#74、#78→#76、#79→#78 | encoder単独全12条件不合格。#73のPow候補は不採用。#79の固定入力post norm単独は合格 |
| native診断 | #75→#60、#77→#75、#80→#77 | native layer / 内部参照のbit再現、FP64投影候補合格。全encoder修正ではない |

API・前処理・定数転送は開発統合候補。#41 / #53 / #68 / #70 / #72〜#80は失敗や診断の記録を含む。CIのgreenを実モデルgateの代わりにしない。依存branchをmainへ渡す際は差分・回帰・受け入れsourceを確認し、mergeは別の明示指示後とする。

## 数値修正

音声はnative exact-inputのpost norm単独がCPU/GPUで合格したため、前段のFP32線形積和と後段での差の増幅を切り分ける。FP64候補はoracleでありRuntimeに採用しない。FP32 block積和＋Neumaier補償和を小さい独立fixtureからTDD比較する。モデル・元golden・`atol=1e-5` / `rtol=1e-4`を維持する。

画像は新旧参照の文脈一致を先に満たし、元の保存graph・入力・重みhashを維持してstage差を局所化する。音声候補の成功だけで画像へ同じ修正を採用しない。

工程2の完了証拠は元画像12条件と、元音声6信号のPython保存export・Sentis CPU/実GPUの固定数値gate。小さいgreen・cosineのみの成功・FP64候補・skipは代替にならない。

## 実入力から最終API

工程2合格後、Texture/RenderTextureとPCMから前処理→encoder→backbone→768次元APIを固定Pythonへ照合する。Query/Document/Rawの採用role、画像全profile、音声mask/長さ境界、例外後再利用、借用/所有権・二重Dispose・既存text15件/検索回帰を確認する。

GPUでは中間readback禁止、要求/実backend一致、計算fallback/flush 0を確認し、profiler証拠を別に残す。最終vectorは有限・単位長、FP32 cosine>=0.999、Float16重み>=0.99。固定参照とsource/model/input hash、完了集計・exit codeを対応付ける。

## 実行環境と今回のledger

ActionsのRepository Secrets / self-hosted runnerは開始時0件。ローカル6000.3.21f1は不在、6000.3.16f1は存在する。21f1のlaunch失敗は環境不足でありTDD redではない。元consumerのdowngrade要求を受け入れず、別consumerを16f1で作成した。初回UPMは`path Received undefined`で失敗し、数値未実行。再起動はこのterminal initialization failureの切り分けとして行う。実際のUnity版と補修条件を必ず記録する。

FP32演算helperはAPI不足13 assertion red→13 green。固定context比較/CLI/producerは5 red（比較API不足3、CLI flag不足1、producer keyword不足1）→既存を含む48 green。workflow条件は1 assertion red/対照1 pass→修正後検証。ローカルPythonは小さいfixtureのみ、tools/uv。実重み比較は専用Actionsへ渡す。

工程1は本snapshotと依存表で整理済み。工程2・3は未完了。各候補の実測をこのledgerへ追記し、失敗や未実行を残す。
