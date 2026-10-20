# Changelog

## 0.1.0 — Release candidate, not published

- Prepare the first Windows Editor / Player release: Sentis GPU text, Texture / RenderTexture and mono PCM embeddings, normalized 768-dimensional vectors, and cross-modal search. Keep the existing source-qualified FP32 / Float16-weight numerical results and the Runtime implementation unchanged.
- Record matched Windows CPU reference audio eight times and image twelve times under explicitly declared per-profile settings, alongside forty GPU media runs and sampled memory observations. Preserve failed configurations, missing samples and measurement limits. CPU references are validation tools and do not add a CPU fallback to the SDK.
- Select the tested Windows / D3D12 / RTX 4070 Ti SUPER, 64 GB RAM / 16 GB VRAM class as the initial support baseline. This does not establish a physical memory minimum or qualify other GPUs or platforms.
- Add explicit Windows audio and image CPU CI routes and a relative native-build-path fix. Local related tests and workflow lint pass; hosted execution, required CI and fresh public-tag consumer acceptance remain pending. The owner deferred removal of old provider-side commit copies; current outgoing history and payload privacy checks remain required.
- Keep Apache-2.0 code licensing and separately obtained, hash-pinned models with their own LICENSE / NOTICE / MODIFICATIONS. Models and validation Player builds are not part of this UPM distribution.

The version field is prepared for release review. No `v0.1.0` tag or formal Release has been published. See `docs/windows-release-checklist.md` and `docs/next-release-plan.md` in the repository for the remaining release gates. macOS / iOS / Android remain future qualification work.

## 0.1.0-pre.1 — Unreleased

### Current checkpoint (2026-10-20)

- Complete Windows-first stages 1–3 for source-qualified candidate `e5b6bd9` (same full tree as recorded candidate `9967516`). Merge PR #113 after all 15 Required CI jobs and 13 numerical archives pass independent readback. Actual Windows Editor runs cover both weight formats, all media roles, mixed search, all 15 original text cases per format and RenderTexture roles; accepted processes exit normally with no computation CPU fallback or intermediate readback. Re-audit the existing Release IL2CPP / High Player results against unchanged SDK inputs. Keep failed test attempts and historical execution sources separate. Performance, minimum hardware, fresh-tag installation and formal publication remain incomplete.

- Adopt the user-approved Windows-first formal release scope: Windows Editor / Player, all three input modalities, cross-modal search, both accepted weight formats and audited GPU execution. Measured performance, minimum hardware, fresh-tag installation and formal publication remain required. macOS / iOS / Android qualification is future work; existing failures and missing execution remain recorded separately.

- Document the published FP32 / Float16-weight six-module model previews and the explicit, hash-pinned acquisition path. Fixed-source Windows verification covers anonymous empty-cache acquisition, interruption / recovery, warm reuse, corrupt-object repair and GPU execution. Keep model LICENSE / NOTICE / MODIFICATIONS with the separately obtained bundle. Acquisition is distinct from regenerating models from the upstream checkpoint.
- Historical source `f0fc2a7` has Windows D3D12 / IL2CPP Release / High-stripping mixed Player acceptance in both weight formats across four independent cold / warm processes. Later Linux and Windows audio CPU reference chains retain their own execution sources and unchanged original numerical conditions. These source-specific records retain their original scope; matched CPU/GPU performance, minimum hardware, future macOS / iOS / Android acceptance, formal SDK version / tag / Release and fresh-tag consumer remain incomplete.

### Earlier implementation checkpoints

- Align Package Manager display name, description and keywords with the implemented text, image and audio APIs. Package ID, development version and validation scope are unchanged.

- Integrated PR #31: fresh fixed-Git consumer and Windows text-search IL2CPP Release / High-stripping Player acceptance, independent cold / warm FP32 GPU runs and normal exits.
- Integrated PR #83: importable Image and Audio Search sample, audited six-module preparation and cache, GPU cancellation / retry / switching / disposal, async audio-preprocessing cache retirement.
- Actual Windows media validation: six image and six audio cases across five profiles, mixed text-query search, rendered PlayMode UI, and 505 canonical EditMode tests. These source-qualified results do not qualify other devices or multimodal Players.
- Refresh user documentation and roadmap; fresh-checkout multimodal model preparation, manual interaction, remaining sample UX, performance and platform qualification remain open.
- 2026-10-16 integrated runtime snapshot: actual Texture / PCM inputs reach normalized768 GPU vectors for all18 image and18 audio role conditions. Original image and audio encoder gates retain learned coefficients, original references and strict tolerances. Source-specific evidence is retained in `docs/results`; these changes are integrated through PR #79; no release has been published.
- Add staged `PartitionedAudioFeatureEncoder.EncodeAsync` and `AudioEmbedder.EmbedAsync`, with GPU-complete fences, completed cache retirement, owned intermediates, overlap rejection, cancellation and recovery. Only the final vector is downloaded; no CPU inference fallback. The failed synchronous12 GiB guard stop is retained separately.
- Add an Editor-only exit/reload hook for the pinned Sentis2.6.1 shared64-byte buffer cleanup. Tested normal shutdowns report no Persistent warning; long-term allocation stability remains unqualified.
- Actual FP32 / Float16 text regression and related API contracts pass189 tests. Required CI now passes the full six-signal pinned original acceptance; fresh CPU comparison failures remain separately recorded diagnostics. Multimodal Player, other devices, performance and release qualification remain subsequent work.

Earlier implementation checkpoints follow; their original counts and scope describe the state when each change was introduced. Current acceptance is summarized above and in `docs/status.md`.


- Add GPU-only 8-bit Texture / RenderTexture preprocessing with official bicubic antialias rounding, RGB patch order, position metadata and owned output tensors. Image encoder integration and final embedding API remain in development.

- Add owned CPU / GPU PCM preprocessing outputs and Worker reuse / disposal contracts; audio encoder and final embedding integration remain in development.

- 固定16kHz mono PCM入力のAudioPcmInputを追加。161〜179200 samplesと振幅[-1,1]を監査し、owned snapshot・右zero padding / prefix mask・公式frame境界を準備。25 red → 25 green。GPU前処理・encoder・最終音声embeddingは別の未完了gate。

- 固定profileのproject済み画像 / 音声featureを扱うMultimodalBackboneを追加。token / mask / feature数・Model入出力を監査し、caller所有Tensorを借用して最終768次元だけをコピー。21 redから契約22 green、小さい実GPUを含め23 green、破棄featureの追加1 redから最終24件＋既存38件の62 green。実encoder・最終実モデル・GPU前処理・prompt / sample・新Git consumer / Playerは別の未完了gate。

Historical text-only checkpoint (2026-10-10):
- 2026-10-10の検証状況を更新。PR #30まで統合済み、固定Git consumer `fc66af7`の実モデル4条件・UI / cacheを確認。Androidは実APK・署名 / payload監査までで実機は未実行。draft PR #31では通常表示sampleのGUI / Float16 GPU全query・解放が成功した後に終了クラッシュを観測し、全体受け入れは未達。以下の各変更時点の未実行記録は当時の履歴として保持する。
- GUI style測定テストもEditor所有fontを共有し、fixtureのCleanupでIMGUI cacheへ破棄済みfontを残さないように修正。元のglyph境界assertionを維持し、1 failed → 1 passed、sample全67 passed、未準備UI 3回の停止前後Console 0を確認。PR #29統合SHAの新規Git導入・実モデル4条件・UI / file URL cacheは別に合格。最初のEditor検索DBエラーの原因は未確定。
- TextSearch sampleのEditor fontをEditor内で1つだけ所有し、Play / domain reload後も同じnative fontを再利用。修正前2件の失敗と修正後2件・sample EditMode 66件 / PlayMode 2件、Windows実Float16 GPU日英検索・空入力・解放、正常Editor終了時の所有font破棄を確認。新helperのPlayer実行・他OSとnative allocation診断の原因は未検証。
- 検証済みテキストRuntimeをUPMへ移行。公開API・Runtime assembly名・既存ソースとGUIDを維持。
- モデル不要の29件のEditor契約テストを配布パッケージへ移行。
- Sentis 2.6.1 / Newtonsoft JSON 3.2.2の直接依存を明示。
- モデル、計測fixture、開発用自動操作ツールは配布対象に含めない。
- Windows Unity 6000.3.16f1の別consumerでローカルフォルダ依存の解決・compile・契約29件成功。元の検証プロジェクトで実モデルCPU / GPUCompute回帰も成功。
- 2026-10-10にPR #6をmainへ統合。初期UPM検証はローカルフォルダ依存で、Git URLの後続実測は以下に記録。公開tagとmacOS / iOS / Androidの受け入れは未完了。
- PR #10で固定文書のcosine検索、同点ID順、所有リソースを解放する検索sessionをmainへ統合。PR #11でText Searchサンプル、明示的なモデル準備・hash監査・配置・更新手順と、検証済みキャッシュの再利用を追加。
- Windowsの元プロジェクトと新規consumerでfp32 / Float16重み × CPU / GPUComputeの固定全順位・ベクトル一致、CLI 4 passed / skip 0と日英画面操作を確認。
- PR #20で日英表示に必要なfont生成・レイアウトと操作確認を追加。font警告とallocation Logの由来は追跡中。
- PR #24で選択した重みとtokenizerだけを保持する所有cache、ローカルfile / jar URL配置、完全hash、worker使用中lease、キャンセル時の転送解放を追加。成功後1組 / 更新中最大2組のcacheを保持。実Windows file URL / GPU / warm cacheを確認した。実APK / 実機とsample Player画面は未検証。
- PR #25の記録では、PR #24統合SHA `01f3d8682a97053bf4583e14fd8fbed14e2c8cae`を新しい空のWindows Git consumerで導入。UPM 38 / sample EditMode 64 / PlayMode 2、実モデル4条件・全順位 / score、日英入力 / 空入力 / 解放 / warm cache / Play停止を確認。PlayMode CLI接続失敗は同じ実行のXMLで2 passedを回収した。
- Windows検証専用PlayerはPR #23でIL2CPP / compiler Release / High stripping、batch mode / Direct3D12の実buildと実モデル4条件・独立2起動を確認。検証assemblyのpreserve-all条件であり、任意consumerのstripping成功や配布sampleの通常表示 / shutdownへ一般化しない。
- versionは `0.1.0-pre.1`のまま。macOS Editor / iOS / Androidの必須実測、残る安定性、候補 / tag固定導入、正式Releaseは未完了。
