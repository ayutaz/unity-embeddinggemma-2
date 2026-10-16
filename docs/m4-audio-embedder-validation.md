# 最終音声APIのTDDと合成検証

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

2026-10-13。`AudioEmbedder` / `IAudioEmbedder`と`MultimodalPromptBuilder.BuildAudio`を実装した。公式audio marker / prefix → PCM前処理 → encoder / projector → audio-only共通backboneを合成し、独立した768次元配列を返す。[使用例](../Packages/com.ayutaz.embeddinggemma/Documentation~/audio-embedder.md)、[固定sourceのreceipt](results/m4-audio-embedder-composition-windows-20261013.json)。これは実モデル精度の完了ではない。

最初にstage interfaceの仕様とテストを固定Git `a09968a…`へ保存した。interfaceには振る舞いを実装せず、Runtimeのfinal APIは未実装。既存consumerのrequested / resolved一致、compile error / warning 0を確認して実行し、20 assertion failed / passed 0 / skip 0 / CLI exit 1。全failure messageはfinal API未実装だった。

最小実装の固定Git `bf3cd0a…`へ更新後、20 passed / failed 0 / skip 0、CLI exit 0、compile error / warning 0。公開APIの直接呼び出しへ整理した固定Git `9c6f794…`でpackage assembly全体185 passed / failed 0 / skip 0を確認した。Runtime overlayなし、EditorはPID 16260の1つを再利用。各結果を後続headへ付け替えない。公式Test Framework 1.6.0 embeddedの条件を維持する。

小さいSentis graphでCPU / 実GPU、image count=0、audio soft token count=1、公式marker / BOS / EOS、有限768次元・単位長と独立した期待値を照合した。encoder / backboneの途中失敗時に所有した中間結果を解放し、失敗後の再操作が可能なことをtracking adapterで確認した。facadeのDisposeでは借用workerを破棄せず、呼び出し側で引き続き利用できる。GPUの中間Tensorは借用のまま受け渡し、最終vectorのみCPUで受け取る。backboneの実行layerのCPU fallbackも検査した。

実encoderの元6信号・新旧固定参照の不合格はPR #60へ保持し、許容差を変更していない。独立した最終音声Python参照、実前処理 / encoder / zero-image backboneの全chain数値照合、Float16、検索 / 再生 / 明示収録sample・Player・必須実機・最終空consumerは未完了。小さい合成の185 greenをM4全体の成功へ流用しない。
