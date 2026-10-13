# M4: 固定PCM入力境界のTDD検証

更新日: 2026-10-13。A2の固定profile入力準備として `AudioPcmInput`をRuntimeへ追加した。**前処理Worker・音声encoder・最終768次元APIの実装や受け入れではない。** 入力境界の契約を先に固定し、GPU前処理Runtimeへ渡す。

## 入力と所有権

16kHz / mono、161〜179200 samples、finiteで振幅[-1,1]を受け付ける。対応外rate / channel・長さと不正振幅は明示エラー。caller配列のowned snapshotを検査し、固定179200長の右側をzero padding、同じ長さのint32 prefix sample maskを作る。利用側にはReadOnlySpanで公開し、caller配列の変更や公開コピーの変更はowned inputへ影響しない。プロファイル上限は11.2秒、mel frame総数は1119。

`ValidFrameCount`は公式semicausal maskが参照するsource index 160 / 320 / 480 / ...に合わせる。161→1、319 / 320→1、321→2、479 / 480→2、481→3、179199 / 179200→1119を固定期待値として検査する。Pythonの既存 `test_frame_end_mask_and_right_padding_match_official_at_sample_boundaries`は公式Gemma4AudioFeatureExtractorへの独立照合を行っており、その境界契約を起点とする。

## RedとGreen

新しいRuntime / unit sourceを既存consumerのEditor test assemblyへ配置し、未実装stubで25 failed / passed 0 / skipped 0 / inconclusive 0。初回CLIの結果保存先がなく保存エラーとなったが、同じoperationの保存済みNUnit XMLを読み戻して全25件の意図したassertion failureを確認した。重複実行せず、保存エラーをTDD redへ数えていない。

最小実装後はcompile成功、25 passed / fail 0 / skip 0 / inconclusive 0、2026-10-13 01:31:20 UTCに完了。入力長・sample / frame境界、padding / mask、null・rate / channel・NaN / Infinity・範囲外振幅、copy ownershipを検証。[source / XML・結果hashとscope](results/m4-pcm-input-windows-20261013.json)。

この段階はmanaged入力準備のTDDであり、GPU実行・配布Runtime assembly・音声encoder合格ではない。次は固定Git Runtimeへの導入を確認し、A3の前処理Worker・GPU tensor / maskの所有権と受け渡しを実装する。downmix / resampleを追加する場合も別の独立参照とredから始める。正式Release / 公開tagは作成しない。
