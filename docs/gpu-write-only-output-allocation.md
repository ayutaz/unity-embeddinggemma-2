# GPU出力tensorの不要なCPU確保・転送を省く

2026-10-18。候補SDK `d7b2233` を性能branchへ `1397a88` として取り込んだ。[source対応](results/gpu-output-integration-source-correspondence-20261018.json)でSDK packageのGit treeが実行済みsnapshotと一致することを確認している。最終candidate全体の合格ではない。

## 変更と根拠

9種類のGPU演算で、shaderが全要素を書く出力を、CPU storageを持たないfloat tensorとして作る。既存の `ComputeTensorData.Pin` が新しいGPU storageを確保し、その後は同じshader・dispatch・GPU fence・cache解放を使う。入力PCMや係数の明示的な転送は従来どおりで、演算順序・許容差・GPU要件は変えていない。

固定Sentis 2.6.1では、shape-only constructorの既定動作はCPU storageの確保とゼロ初期化であり、Pinはその内容をGPUへ変換・転送する。全要素を上書きする出力に、このCPU準備は不要である。部分書き込みや未初期化の値を入力として使うtensorへ一般化しない。

## 検証済みの範囲

- [TDDとGPU演算回帰](results/gpu-write-only-output-allocation-tdd-20261018.json)：CPU storageが残るscaffoldで3件中2件Red→3件Green。実GPUで既存の演算・所有権・tiny graph等の回帰120 pass / fail0 / skip0 / exit0。実モデル全体の代用にはしない。
- [同一FP32画像の独立比較](results/gpu-output-square-fp32-matched-comparison-20261018.json)：旧SDK `e28a5fd` と候補 `d7b2233`、同じ計測 `c2f766e`、square-0・model descriptor・GPUを固定。各4回が元のPython参照・解放・正常終了を満たし、全4出力のbyte hashが前後で一致した。各回モデル180区間・演算179区間のGPU値も欠測なし。
- 初回後3回の計測付きAPI経過時間は、修正前19.689 / 19.299 / 29.389秒、修正後5.214 / 4.832 / 5.481秒だった。これはこの固定画像の観測であり、通常API・他profile・平均性能・一般的な改善率を保証しない。
- [Float16重みの独立比較](results/gpu-output-square-float16-weights-matched-comparison-20261018.json)も同じsquare-0・計測・GPUで各4回の参照一致・解放・exit0と全4出力の前後byte一致を確認した。初回後3回は修正前34.048 / 32.479 / 30.204秒、修正後5.067 / 5.009 / 4.603秒。こちらも計測付き固定入力の観測であり、形式間の速度比較や通常APIの保証ではない。
- shader ResourcesのGit treeは旧SDKと一致する。重み・graphを再生成していない。

別のproject/processで順に実行し、OS・shader cacheや他アプリの負荷は制御していない。Profiler/markerの影響があり、CPU送信区間の経過時間とGPU queue区間は重なり得るため、両者を足したり差し引いたりしない。

## 残る受け入れ

新SDKの他画像profile・音声・テキストの元参照、混在検索の全順位・score・キャンセル再試行・解放を確認する。通常APIの対応する入力と条件での計測、host / GPU memory、残るstage GPU時間・非同期割り当て・Editor測定も必要である。旧SDKの全40回やFloat16全6 moduleの合格を、新SDKの合格へ流用しない。

PR #87はDraftを維持し、最新headのCIも別に確認する。公開empty-cache取得・許可後の実マイク・最終candidate操作を含む優先1〜3全体は未完了である。
