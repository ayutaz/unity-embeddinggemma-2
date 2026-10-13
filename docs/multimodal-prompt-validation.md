# 公式マルチモーダルpromptのRuntime契約

2026-10-13。共通基盤C3のprompt / mask構築をTDDで追加する。`MultimodalPromptBuilder`はtokenizer JSONとtoken設定、固定backbone profileを受け取り、単一画像・単一音声またはその組合せの完全なpromptをtokenizeする。encoderの実行やfeature tensorのreadbackは行わない。

画像placeholderは実feature数256 / 252等に合わせ、公式の開始 / 終了markerで囲む。音声placeholderは公式processorと同じ2段stride-2 mask縮約の有効数から作る。query / document prefixとtitleは既存text契約を維持。元のtext用tokenizerに保存された128長等のtruncation / paddingを解除し、全token化後に固定profileへ右paddingする。上限超過は推論前に拒否し、切捨てはしない。token IDの設定違い、不正mask・欠落 / 重複placeholder・展開済みmarkerを拒否する。結果のIDs / maskは独立したcopyとして取得できる。

## TDDの証拠

Runtime APIの未実装を24 assertion redで確認した。CLIのstdout保存は保存先未作成で失敗したため、同じ実行のNUnit XML `20261013_052022_9400321_e51d38342f5740eda701ad8f27d264fc.xml`から24 failedを回収し、再実行しなかった。初回green試行の検証専用assemblyにはTokenization参照が不足しcompile失敗。その環境失敗をredに加算せず、参照を補った後、2026-10-13 05:22:48 UTCに24 passed / failed 0 / skipped 0 / compile成功を確認した。これは検証専用prototype assemblyの結果で、配布Runtime assemblyの結果ではない。

Pythonの公式fixture生成・固定source監査・重みを取得しないCLIは4 assertion redから追加した。初回の関連回帰はWindowsのテスト側でUTF-8を指定しない読み込みに失敗し、20 passed / 1 failed。読み込みを修正し関連21 passedを確認した。固定revisionのtokenizer / processor設定とquery / document prefixは既存cacheの小さい設定だけで確認した。

## 独立参照と次の受け入れ

Actionsの`prepare-prompts`はmodel weightsを取得しない。固定revisionの4設定fileのhash、source / library版と、公式processorのimage / audio置換・HF tokenizerのIDs / maskを15ケースに保存する。画像256 / 252、音声の1 / 4 / 5 / 8 / 9 / 17 / 2999 frame境界と穴のあるmask、両モダリティの順序、query / document・日本語title、空rawを含む。大きいtokenizerはartifactに保存しGitへ含めない。

実tokenizerの受け入れは`MultimodalPromptReferenceTests`で固定source・全file hash・15ケースの文字列 / IDs / maskを照合する。型のassemblyは`EmbeddingGemma.Runtime`を要求し、prototypeを配布結果としない。次にprototypeを退避し、固定Git packageへ更新してpackage契約・この公式参照を実行する。これらは実encoder・最終embedding・GPU連結・sample / Player・他OSの成功とは別に判定する。正式Release / tagは作成しない。
