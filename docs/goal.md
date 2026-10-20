# プロジェクトのゴール

更新: 2026-10-20。ユーザーの方針により、最初の正式リリースはWindows Editor / Playerを対象とします。macOS / iOS / Androidは今後の対応とし、未検証の結果を合格に読み替えません。実際の対応範囲は[検証状況](status.md)、作業順と残件は[実行計画](next-release-plan.md)を参照してください。

## 目指すもの

**Unity Sentisで、テキスト・画像・音声の埋め込みをGPU上で作り、モダリティをまたいだ検索を利用できるようにする。**

1. **Sentisで推論する**：追加のllama.cpp・ONNX Runtime・LiteRT等のネイティブ推論プラグインを使わない。Runtimeの直接依存を明示する。
2. **3種類の入力を同じ空間へ変換する**：画像・音声encoderの特徴量を共通backboneへ渡し、正規化された768次元vectorを返す。text→text / image / audioの検索と混在索引を提供する。
3. **GPU上で前処理と推論をつなぐ**：Texture / RenderTextureの画像変換、PCMからの音声特徴量生成、encoder・backboneをGPUで実行する。中間tensorのCPU readback・計算fallbackを禁止する。

テキストtokenize、ファイルdecode・音声取得 / uploadはCPU側の入力境界、最終768次元配列のCPU取得は出力境界として認め、転送を記録します。GPU非対応をCPUで代替して合格にはしません。

## 達成基準

| 項目 | 基準 |
| --- | --- |
| FP32最終vector | 有限768次元・単位長、固定Python参照とのcosine >= 0.999 |
| 採用Float16重み形式 | 同じ条件でcosine >= 0.99。全演算FP16化や高速化を自動的に保証しない |
| encoder・中間値 | 固定参照と元の要素別許容差で照合。後から期待値・許容差を失敗へ合わせない |
| 横断検索 | 固定queryと文書・画像・音声の全順位・score、混在索引・同点処理を照合 |
| GPU | actual backendとfallback・readbackを実GPUで監査。skip・tiny graph・import成功・Python CIだけでは合格にしない |
| 導入・運用 | 空consumerの固定Git導入、モデル準備、sample操作、切替・キャンセル・解放・回復を再現 |
| 初回正式リリースの必須環境 | Windows Editor / Player。CPU比較と実GPU受け入れを分離。macOS Editor、iOS / Android Player実機は今後別途受け入れ |
| 性能 | 同一source・モデル・仕事量で初回 / warm・各stage・メモリ・正常終了を測定。最低要件は実測から決める |

参照は固定revisionのtransformers / sentence-transformers実装に合わせます。prompt、512→768 projection、pooling、L2正規化、次元・mask・feature配置を揃えます。固定入力の一致と任意入力の意味検索品質は別の評価です。

## 成果物と到達点

| マイルストーン | 現在の到達点 | 続く条件 |
| --- | --- | --- |
| M1 テキスト本体 | Windows Editorで推論・保存・量子化・固定入力を受け入れ済み | 環境・profile変更時の再照合 |
| M2 UPM・検索・検証基盤 | UPM、text sample、固定Git consumer、Windows text Playerは検証済み | 最終Windows candidate・系統的性能測定・新規導入 |
| M3 画像 | GPU前処理・元encoder・全role最終APIと画像音声検索sampleは固定sourceで合格 | 最終Windows candidateの利用者向け準備・UX・Player・測定 |
| M4 音声 | 元encoder・実PCMの全role最終API、async前処理、cache・混在検索sampleは固定sourceで合格 | 最終Windows candidateの音声再生・明示収録・準備・Player・測定 |
| 正式公開 | 未実施 | 全必須条件→最終candidate / tag導入→Release |

UPMコード、モデル変換用Python、検索サンプル、環境別の速度・メモリ結果を提供します。モデル重み・生成model・大きい検証artifactはGit / LFSに含めず、取得・変換手順を提供します。リポジトリのpublic公開と正式リリースを区別します。

Windowsの正式公開後にmacOS / iOS / Androidのbuild・実モデルGPU・実機受け入れを進めます。他環境のライセンス・署名・端末不足はWindows先行公開の条件に含めず、対応環境としても表示しません。

## 対象外

- 動画の埋め込み、モデル学習・fine-tuning、本格的なベクトルDB。
- ネイティブ推論プラグインを使った代替実装。
- Web / WebGPUの正式公開必須条件への追加（余力があれば別途検討）。

開発は[AGENTS.md](../AGENTS.md)に従い、TDD・uv・PR運用を維持します。文書や小さいテストの成功を実モデル・実機の成功として扱いません。
