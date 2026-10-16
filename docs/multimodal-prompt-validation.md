# 公式マルチモーダルpromptのRuntime契約

> 文書整合確認: 2026-10-14。[現在の状態](status.md) / [GPUを主経路とする再開計画](multimodal-tdd-plan.md)。本文の数値・source・実行中／次の作業は記載当時の記録。現在headの実行結果として扱わない。

2026-10-13。共通基盤C3のprompt / mask構築をTDDで追加する。`MultimodalPromptBuilder`はtokenizer JSONとtoken設定、固定backbone profileを受け取り、単一画像・単一音声またはその組合せの完全なpromptをtokenizeする。encoderの実行やfeature tensorのreadbackは行わない。

画像placeholderは実feature数256 / 252等に合わせ、公式の開始 / 終了markerで囲む。音声placeholderは公式processorと同じ2段stride-2 mask縮約の有効数から作る。query / document prefixとtitleは既存text契約を維持。元のtext用tokenizerに保存された128長等のtruncation / paddingを解除し、全token化後に固定profileへ右paddingする。上限超過は推論前に拒否し、切捨てはしない。token IDの設定違い、不正mask・欠落 / 重複placeholder・展開済みmarkerを拒否する。結果のIDs / maskは独立したcopyとして取得できる。

## TDDの証拠

Runtime APIの未実装を24 assertion redで確認した。CLIのstdout保存は保存先未作成で失敗したため、同じ実行のNUnit XML `20261013_052022_9400321_e51d38342f5740eda701ad8f27d264fc.xml`から24 failedを回収し、再実行しなかった。初回green試行の検証専用assemblyにはTokenization参照が不足しcompile失敗。その環境失敗をredに加算せず、参照を補った後、2026-10-13 05:22:48 UTCに24 passed / failed 0 / skipped 0 / compile成功を確認した。これは検証専用prototype assemblyの結果で、配布Runtime assemblyの結果ではない。

Pythonの公式fixture生成・固定source監査・重みを取得しないCLIは4 assertion redから追加した。初回の関連回帰はWindowsのテスト側でUTF-8を指定しない読み込みに失敗し、20 passed / 1 failed。読み込みを修正し関連21 passedを確認した。固定revisionのtokenizer / processor設定とquery / document prefixは既存cacheの小さい設定だけで確認した。

## 独立参照と次の受け入れ

Actionsの`prepare-prompts`はmodel weightsを取得しない。固定revisionの4設定fileのhash、source / library版と、公式processorのimage / audio置換・HF tokenizerのIDs / maskを15ケースに保存する。画像256 / 252、音声の1 / 4 / 5 / 8 / 9 / 17 / 2999 frame境界と穴のあるmask、両モダリティの順序、query / document・日本語title、空rawを含む。大きいtokenizerはartifactに保存しGitへ含めない。

実tokenizerの受け入れは`MultimodalPromptReferenceTests`で固定source・全file hash・15ケースの文字列 / IDs / maskを照合する。型のassemblyは`EmbeddingGemma.Runtime`を要求し、prototypeを配布結果としない。次にprototypeを退避し、固定Git packageへ更新してpackage契約・この公式参照を実行する。これらは実encoder・最終embedding・GPU連結・sample / Player・他OSの成功とは別に判定する。正式Release / tagは作成しない。

## 固定Git導入と環境回復

Runtime / 単体テストのprototypeをAssetsから退避し、固定Git `8a7d2f0dea30be95a3158830fb0d9c9de895efc4`へ更新した。初回のCLIは未対応`--timeout-seconds`を引数解析で拒否し、Editorへ要求されていない。対応オプションでの最初のUPM要求は受理されたが、serverはHTTP 500を返した。Editor内の`ALLUSERSPROFILE`が欠け、`ProgramData`等は設定済みだった。Editorのプロセス内で`ALLUSERSPROFILE`を補い、親PIDと実行fileを照合した該当UPM serverだけを停止した。Editor PID 16260は保持し、次のUPM server PID 173168でrequested / resolved SHAが一致した。観測timeoutだけを理由とする再起動・再要求ではない。次回のEditor起動前にはPowerShellで`$env:ALLUSERSPROFILE=$env:ProgramData`を指定する。

2026-10-13 05:38:24 UTCに配布packageの全142 passed / failed 0 / skipped 0 / compile成功を確認。既存consumerを更新した条件であり、空consumerの最終candidate再現ではない。公式Test Framework 1.6.0のembedded条件を維持する。source監査は119 Git fileのうち118件がcanonical改行で一致し、package.jsonはUPMの`_fingerprint`だけを除いてsemantic一致。最初の監査コマンドはtools cwdでtreeが空だったため、0件をsource証拠として受け入れず、Git root指定と非空guardで修正した。raw bytes全件一致とは表現しない。

[PR #55のCI 38351727099](https://github.com/ayutaz/unity-embeddinggemma-2/actions/runs/38351727099)はhead `8a7d2f0`の全8チェック成功。公式prompt artifactのproducerはmerge ref `dc185218d6b682f53b191e21600255be19c380e7`であり、PR headとは別。15ケースを取得し、固定source / revisionと全file hashを照合した。実tokenizerのRuntime照合は、この取得済みartifactと固定Gitを使う同じEditorで実行する。

2026-10-13 05:43:27 UTCに実tokenizer照合が1 test passed / failed 0 / skipped 0 / CLI exit 0で完了し、内部の全15ケースが文字列・token ID・attention maskの完全一致を満たした。型のassemblyは`EmbeddingGemma.Runtime`、Runtime source overlayなし。[固定GitとTDDのreceipt](results/multimodal-prompt-windows-20261013.json)。full tokenizerの反復初期化とfile監査を含むharnessの所要時間であり、warm prompt構築・モデル推論のbenchmarkとはしない。これ以降の文書commitへ実測sourceを付け替えない。

次はこのpromptを使い、実encoderの受け入れ済みfeatureを対応するstatic backbone profileへ渡す。現在の合成本体検証の280 / 280 feature profileを256 / 252画像や可変音声maskへ流用せず、profile別exportと最終768次元の独立参照をTDDで準備する。画像FP32 / 音声encoderの不合格と全体計画の残るgateは保持する。
