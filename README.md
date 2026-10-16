# unity-embeddinggemma-2

Unity の推論ライブラリ [Sentis](https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/index.html) だけを使って、Google の [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) を動かすプロジェクトです。テキスト・画像・音声を、同じ 768 次元の空間の埋め込みに変換できるようにします。

> **2026-10-16: PR #81で実GPUの画像・音声encoderと最終APIを検証済みですが、手順4は未完了です。** 固定sourceのWindows Editorで、元画像6件・3 profileは778,240値がbit一致、元音声6信号・2 profileは824,320値で元の許容差超過0でした。実Texture / PCMから正規化768次元までのQuery / Document / Rawは各18条件に合格し、計算・非空feature入力のCPU fallbackとAPI中間readback /flushは0です。RenderTexture、入力境界、所有権・破棄、例外後の復帰、テキスト回帰も確認しています。[確認済み結果と残件](docs/status.md) / [公開API](docs/runtime-api.md)。
>
> これらはUnity6000.3.16f1 /Sentis2.6.1 /RTX4070 Ti SUPER /D3D12のsource付き結果です。最新headで全条件を再実行したという主張ではなく、main統合・正式リリースも未実施です。必須CIは音声ONNX比較で失敗し、CPU参照基準の方針確認とnoiseのCPU ONNX最終値取得が残っています。実GPU結果とCPU変換の診断は別に扱います。過去の不合格も[固定記録](docs/gpu-execution-and-silu-validation.md)と`docs/results/`に保持しています。

> **状態: Windows EditorのM1受け入れ検証完了**
> 2026-10-09: PR #1〜#4はmain `8146107`へ統合済み。[PR #4](https://github.com/ayutaz/unity-embeddinggemma-2/pull/4)で`.sentis`保存・再読み込み、Float16重み量子化、性能・メモリ測定まで実装・検証しました。
> tokenizerの全15件一致、fp32 / Float16重みのCPU / GPUCompute各15件、公開C# API、C#単体契約30件が合格。M1統合mainのPython CIは4環境各64件が合格しました。
> GPU定常推論の中央値はfp32 50.26ms、Float16重み58.20ms。保存サイズは約1.096GBから0.554GBへ減りました。測定条件と限界は [検証記録](docs/m1-completion-validation.md) を参照してください。
> クラウドUnity CIはSecretsが必要なLinux CPUの手動補助検証です（未実行）。
> リポジトリは public ですが、UPM パッケージのリリースはまだ行っていません。

> **2026-10-10: M2はWindowsの固定Git導入と実モデル検索を確認済み、正式リリースは未完了です。** テキストRuntimeは `com.ayutaz.embeddinggemma` の開発版 `0.1.0-pre.1`としてmainへ統合済みです。
> 全PRでCIとパッケージ監査を実行し、mainへの変更はPRとRequired CIを必須にしています。確認基準main `b1a8c67`（PR #25後）の[CI run 38023036522](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38023036522)は全8 job成功です。
> 新しい空のWindows consumerへPR #24統合SHA `01f3d8682a97053bf4583e14fd8fbed14e2c8cae`をGit URLで導入し、SHA解決・source照合・compile、UPM 38 / sample EditMode 64 / PlayMode 2、実モデルFP32 / Float16 × CPU / GPUComputeの4条件と全順位・score、日英画面入力・空入力・解放を確認しました。[導入と実測](docs/m2-latest-git-consumer-validation.md)を参照してください。
> 選択モデルだけを保持するsample cacheはfile URL初回展開・warm完全hash・実GPU検索を確認済みです。Windows検証専用PlayerもIL2CPP / Release / High strippingのbuildと実モデル4条件・独立2起動を確認しました。[sample cache](docs/m2-sample-cache-validation.md) / [Playerの条件と結果](docs/m2-player-il2cpp-validation.md)。
> この段落は2026-10-10の到達点です。後続のWindows Player実測・環境補修条件・残タスクは[現在の状態](docs/status.md)を参照してください。

> **正式リリースは画像・音声を含む全対応と必須検証の完了後に行います。** M2の基盤安定化 → 共通マルチモーダル基盤 → M3画像 → M4音声 → 全対象環境の検証 → 配布準備の順です。全実装をTDDで進めます。[全体計画](docs/multimodal-tdd-plan.md) / [M3のRed・Greenと実モデル条件](docs/m3-plan.md) / [M4のRed・Greenと実モデル条件](docs/m4-plan.md)。画像・音声の検索sample、反復性能・メモリ、Windows Player・他OS /実機、配布準備は引き続き後続工程です。PR #81のGPU数値検証・最終API結果と必須CIの残件は上記の最新状態を参照してください。

## ゴール

次の 3 つをすべて満たすことを目指します。詳しくは [docs/goal.md](docs/goal.md) を参照してください。

1. **Sentis だけで動かす**: llama.cpp や ONNX Runtime などのネイティブプラグインを使わない
2. **テキスト・画像・音声に対応する**: 3 つの入力を同じ空間の埋め込みにして、入力の種類をまたいだ検索ができる
3. **入力を GPU 上で処理する**: `RenderTexture` やマイクの音声を、CPU に読み戻さずに埋め込みにする

## マイルストーン

| # | 内容 | 状態 |
| --- | --- | --- |
| M1 | テキスト用モデルを Sentis で動かす | Windows Editor受け入れ検証完了（保存・量子化・測定を含む） |
| M2 | テキストUPM・サンプル・検証基盤を安定化する | Windows実測あり。標準環境再現・新candidate・性能・他環境は未完了 |
| M3 | GPU画像前処理・encoder・共通本体・画像検索 | PR #81の固定sourceで画像6件 /3 profileと実Texture /RenderTextureの全role合格。main統合・sample・Player等は未完了 |
| M4 | GPU音声前処理・encoder・共通本体・音声検索 | PR #81の固定sourceで元6信号 /2 profileと実PCMの全role合格。CPU ONNX /必須CI・main統合・sample・Player等は未完了 |
| 最終公開 | 全対応と必須環境の検証後に正式リリース | 未着手 |

## 動作環境

| 項目 | バージョン |
| --- | --- |
| Unity | 6000.3.16f1(URP 2D テンプレート) |
| Sentis(`com.unity.ai.inference`) | 2.6.1（ローカル Editor で依存解決・コンパイル済み） |
| Unity CLI Loop | パッケージ 3.14.0 / dispatcher 3.8.1 / project runner 3.8.0 |
| Python(モデルの変換用) | 3.13 以上、[uv](https://docs.astral.sh/uv/) で管理 |

## フォルダ構成

```
.
├── Assets/          Unity のアセット
├── Packages/        Unity のパッケージ設定・EmbeddingGemma UPM
├── ProjectSettings/ Unity のプロジェクト設定
├── docs/            調査結果と設計のドキュメント
└── tools/           モデルを変換する Python プロジェクト(uv)
```

## Python 環境(tools/)

モデルの書き出しと、正しさを確かめるための参照データの作成に使います。Python は uv で管理します。

```sh
cd tools
uv sync --locked             # lockfile と同じ依存をインストールする
uv run --locked python <スクリプト> # スクリプトを実行する
uv add <パッケージ>          # 依存を追加する(pip install は使わない)
```

主な依存は torch・transformers・sentence-transformers・onnx です。

開発は TDD で行い、重い実行は GitHub Actions を利用します。`main` へ直接 push せず、作業ブランチから PR を作成します。
CI の構成と Unity ライセンスの準備は [docs/ci.md](docs/ci.md)、M1 の詳細手順と進捗は [docs/m1-plan.md](docs/m1-plan.md) を参照してください。

Unity のローカル自動操作とハーネスの導入・実行方法は [docs/automation.md](docs/automation.md) を参照してください。

小さいモデルを使うオフライン単体テスト:

```sh
cd tools
uv run --locked pytest -q
```

GitHub Actions が実行する参照生成・変換コマンド:

```sh
uv run --locked python -m embeddinggemma_tools prepare --output ../artifacts/m1
```

モデルの revision を固定して参照データ・設定済み tokenizer・`.pt2` を生成し、保存後のモデルを参照実装と比較します。
この Python 側の一致と、Sentis 側の M1 合格は別々に検証します。

M1のWindows Editor結果とテキスト検索・固定Git導入の既存実測を維持します。PR #81では元encoderのGPU数値gateと実入力→最終APIを固定sourceで確認済みですが、必須CIは未達です。手順4の残件を解消した後、sample /性能・メモリ /Player・必須環境 /最終空consumerへ進みます。
残作業の順序・依存・完了条件は [M2計画](docs/m2-plan.md)、実装済みAPIは [C# API手順](docs/runtime-api.md) を参照してください。
新しい実装も作業ブランチ / PR を使い、マージは依頼があるまで行いません。
M1の完了条件と当時の測定は [M1計画](docs/m1-plan.md) / [完了検証記録](docs/m1-completion-validation.md)、最新のCIとUPM検証は [現在の状態](docs/status.md)を参照してください。
ローカル検証に GitHub Secrets は不要です。任意の Linux CPU 手動 CI を使う場合だけ準備します。

## UPM開発版の導入

Unity 6000.3.16f1の別プロジェクトで、Package Managerから
[package.json](Packages/com.ayutaz.embeddinggemma/package.json)を「Add package from disk」で指定します。
Sentis 2.6.1とNewtonsoft JSON 3.2.2はパッケージの依存から解決します。
Git URLのcommit固定による導入指定とAPI使用例は [パッケージ文書](Packages/com.ayutaz.embeddinggemma/Documentation~/index.md)を参照してください。Windowsの固定Git導入は既存sourceで実測済みです。最新マルチモーダルcandidateの空consumer導入は未検証です。
モデルは含まれていません。Text SearchサンプルはPackage Managerからインポートできます。[モデル準備手順](docs/model-preparation.md)に沿って明示的にモデルを配置してください。公開tagはまだありません。

## モデルファイルについて

モデルファイル(`.safetensors`、`.pt2`、`.onnx`、`.sentis` など)はリポジトリにコミットしません。Git LFS も使いません。`tools/` の CLI が Hugging Face の固定 revision から取得して変換します。Python export照合はCIで成功し、SentisのWindows Editor CPU / GPUComputeでも固定15件の精度を検証済みです。他環境とUPM配布はM2で確認します。

## ドキュメント

| ドキュメント | 内容 |
| --- | --- |
| [docs/status.md](docs/status.md) | 最新のmain / PR / CI、検証済みの範囲、残タスク |
| [docs/goal.md](docs/goal.md) | ゴール、達成の基準、マイルストーン、対象外のこと |
| [docs/multimodal-tdd-plan.md](docs/multimodal-tdd-plan.md) | M2〜M4と全必須検証から正式公開までのTDD・依存・PR分割 |
| [docs/m3-plan.md](docs/m3-plan.md) | 画像の参照固定、GPU前処理・encoder・検索のRed / Greenと実測条件 |
| [docs/m4-plan.md](docs/m4-plan.md) | 音声の参照固定、GPU前処理・encoder・検索 / 収録のRed / Greenと実測条件 |
| [docs/m1-plan.md](docs/m1-plan.md) | M1 の詳細計画、TDD の進め方、検証状況 |
| [docs/m1-completion-validation.md](docs/m1-completion-validation.md) | M1完了の数値、ソース対応、CI、失敗履歴と測定の限界 |
| [docs/m1-validation.md](docs/m1-validation.md) | 基盤導入当時のTDD・CI・失敗履歴 |
| [docs/runtime-api.md](docs/runtime-api.md) | 実装済みのテキスト推論・保存API |
| [docs/automation.md](docs/automation.md) | uloopの導入、検証ハーネス、モデル成果物の監査・配置 |
| [docs/m2-plan.md](docs/m2-plan.md) | 残作業の順序、UPM・検索サンプル・他環境・配布の完了条件 |
| [docs/m2-search-plan.md](docs/m2-search-plan.md) | 検索サンプルとモデル準備の実装済み範囲・残る受け入れ条件 |
| [docs/m2-search-validation.md](docs/m2-search-validation.md) | TDD・CI・Windows検索4条件の実測と未完了項目 |
| [docs/model-preparation.md](docs/model-preparation.md) | 検索モデルの取得・監査・変換・配置・更新と新規consumer導入 |
| [docs/m2-package-validation.md](docs/m2-package-validation.md) | CI・main保護・UPM移行、新規Unityプロジェクトへの導入結果 |
| [docs/ci.md](docs/ci.md) | GitHub Actions、PR 運用、Unity CI の準備 |
| [docs/technical-approach.md](docs/technical-approach.md) | Sentis 2.6 の調査、EmbeddingGemma 2 の構造、モデルを持ち込む方法の比較、設計案、検証方法 |
| [docs/embeddinggemma-2-unity-novelty.md](docs/embeddinggemma-2-unity-novelty.md) | 新規性の初回調査と2026-10-09の限定再確認 |

## ライセンス

このリポジトリは [Apache License 2.0](LICENSE) で公開します。

EmbeddingGemma 2 のモデルも Apache License 2.0 で配布されています。モデルファイルはこのリポジトリに含めません。
