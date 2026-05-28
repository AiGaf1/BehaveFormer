# Toward A* Continuous Authentication from This Repository: A Candid Research Draft

This document is an internal research draft written in a paper-like style for an A*-level security/biometrics direction. It is deliberately candid about what the repository currently implements, what it does not yet implement, and what would need to change before any submission-ready claims should be made.

## Abstract

This repository implements a research codebase for behavioral continuous authentication centered on keystroke dynamics, with additional multimodal support for inertial measurement unit (IMU) signals. In its current form, the codebase provides: (i) keystroke-only training and evaluation pipelines across AaltoDB, HMOGDB, and HuMIdb; (ii) a separate keystroke+IMU pipeline with spatio-temporal dual-attention style encoders; (iii) embedding-based verification using enrollment/verification protocols; and (iv) evaluation utilities for equal error rate (EER), usability, time to correct reject (TCR), false alarm window interval (FAWI), and false reject window interval (FRWI). The current main keystroke path has also been modernized around PyTorch Lightning, supervised contrastive learning, runtime logging, and an optional Muon-hybrid optimizer path. However, the repository does not yet implement a fully online, attack-start-aware training objective; instead, the current documentation in [`README.md`](README.md) describes a future `TCRAwareLoss` direction that remains largely unintegrated with the active training code. As a result, the codebase is better understood today as a solid research prototype for embedding-based continuous authentication and multimodal behavioral biometrics, rather than as a submission-ready system for time-optimal online continuous authentication. This draft describes the current system faithfully, identifies the methodological and reproducibility gaps to publication, and outlines a concrete roadmap toward a stronger A*-level paper.

## 1. Introduction

Continuous authentication (CA) aims to verify user identity throughout an active session instead of relying solely on a point-of-entry login event. This framing is appealing in settings where a device may already be unlocked when a session hijack occurs, or where a password, PIN, or biometric unlock only guarantees identity at session start. Behavioral biometrics are a natural fit for CA because they can be measured passively during ordinary device use. Keystroke dynamics are especially attractive because typing behavior is frequent, device-native, and plausibly distinctive across users. On mobile devices, IMU signals can further enrich the behavioral profile by encoding device handling, motion, and posture patterns during interaction.

The repository in its current state already moves beyond a toy prototype. Its keystroke path is organized as a shared experiment framework with dataset-specific training scripts, a common model stack, and periodic metric computation over AaltoDB, HMOGDB, and HuMIdb. The multimodal path adds several keystroke+IMU combinations and a distinct encoder family. The evaluation utilities go beyond EER by also computing usability, TCR, FAWI, and FRWI. These are all meaningful ingredients for a CA research program.

At the same time, the current system is not yet enough for an A*-level paper. The dominant training formulation is still embedding-based verification rather than explicit online risk minimization. The public-facing `README.md` currently describes a future attack-start-aware loss instead of the implemented project. The codebase mixes modernized and legacy experiment branches. Some paths show signs of code drift that raise reproducibility concerns. Most importantly, there is not yet a unified manuscript narrative that matches the repository faithfully while also making a sufficiently strong methodological claim.

The contributions of this repository, viewed conservatively and honestly, are therefore threefold:

- The current code already provides a meaningful research platform for keystroke and keystroke+IMU continuous authentication, including shared training infrastructure, modernized keystroke modeling, and multi-dataset biometric evaluation.
- The current implementation still leaves important methodological gaps open, especially around online attack modeling, threshold realism, stronger baselines, and code-path consistency.
- The most promising future research direction is a hybrid continuous-authentication formulation that preserves embedding-based verification while adding per-step risk scoring and an explicit TCR-aware objective.

## 2. Background and Related Work

Behavioral biometrics for CA sit at the intersection of security, biometrics, usability, and sequence modeling. Survey work makes clear that this area cannot be evaluated purely as a static classification problem. For example, *Behavioral biometrics & continuous user authentication on mobile devices: A survey* ([Information Fusion, 2021](https://www.sciencedirect.com/science/article/pii/S1566253520303493)) emphasizes realistic collection protocols, stronger attack models, and the need to move beyond zero-effort evaluation. That perspective is highly relevant here: the present repository contains richer metrics than EER alone, but it does not yet fully train toward the security/usability tradeoffs those metrics are intended to capture.

Among transformer-based behavioral authentication systems, the closest architectural reference is BehaveFormer, which proposes spatio-temporal dual-attention transformers for IMU-enhanced keystroke dynamics and evaluates on AaltoDB, HMOGDB, and HuMIdb ([OpenReview / IJCB 2023](https://openreview.net/forum?id=Neh0qm0MA0)). BehaveFormer is especially relevant because this repository shares the same broad problem setting and even echoes the multimodal architectural direction in its keystroke+IMU code. The multimodal encoder blocks in [`experiments/common/modeling.py`](experiments/common/modeling.py) and the manual training flow in [`experiments/keystroke_imu_combined/combined_training.py`](experiments/keystroke_imu_combined/combined_training.py) should therefore be understood in the shadow of that prior art rather than as automatically novel on their own.

TypeFormer sharpens the comparison further by showing that transformer-family models for mobile keystroke biometrics are now a crowded design space, and that strong benchmarking on AaltoDB is possible with carefully designed temporal/channel modules ([Neural Computing and Applications, 2024](https://link.springer.com/article/10.1007/s00521-024-10140-2)). This matters because it weakens a paper strategy based only on “we used a transformer for keystrokes.” Any strong paper from this repository must therefore lean on evaluation realism, multimodal structure, or objective-function novelty rather than transformer usage alone.

On the timing side, TKCA explicitly frames continuous authentication around the challenge of timely decisions under short keystroke sequences and uncontrolled settings ([Cybersecurity, 2021](https://link.springer.com/article/10.1186/s42400-021-00075-9)). That paper is useful conceptually because it shifts attention from final discriminative accuracy toward when a system can decide. The current repository partially acknowledges this framing through TCR, FAWI, and FRWI metrics, but it does not yet optimize for those metrics directly.

AuthentiSense brings a different but equally relevant perspective: scalability and user-agnostic few-shot behavioral authentication on mobile platforms ([NDSS 2023](https://www.ndss-symposium.org/ndss-paper/authentisense-a-scalable-behavioral-biometrics-authentication-scheme-using-few-shot-learning-for-mobile-platforms/)). Compared with that line of work, the current repository is still closer to a user-specific or enrollment-centered verification paradigm than to large-scale user-agnostic authentication.

Taken together, these references place the repository in a clear position. In its implemented form, it is currently closer to an embedding-based biometric verification framework with richer-than-usual CA metrics than to a full time-optimal online CA system. That is not a weakness to hide; it is the key fact that should structure any honest manuscript around this codebase.

## 3. Current Repository System

### 3.1 Experimental entrypoint and overall layout

The main runnable entrypoint is [`run.py`](run.py). It dispatches among three experiment families:

- `keystroke` for keystroke-only experiments,
- `keystroke_imu` for multimodal keystroke+IMU experiments,
- `tl` for transfer-learning style paths.

It also normalizes datasets (`aalto`, `hmog`, `humi`) and routes training, continuation, testing, preprocessing, and metric-generation commands to the corresponding dataset-specific scripts. This gives the repository a useful top-level structure even though the internals are not yet fully unified.

### 3.2 Keystroke-only path

The main modernized keystroke path is built around the shared Lightning infrastructure in [`experiments/common/lightning.py`](experiments/common/lightning.py). That module currently:

- instantiates a model from a factory,
- trains embeddings using supervised contrastive loss by default via [`experiments/common/loss.py`](experiments/common/loss.py),
- compiles the model when available via `torch.compile`,
- logs periodic validation and train metrics,
- checkpoints on `val_eer`,
- supports both `AdamW` and an opt-in `Muon`/`AdamW` hybrid optimizer path.

The current keystroke model family is defined in [`experiments/common/modeling.py`](experiments/common/modeling.py). The main `KeystrokeModel` is not a plain transformer over raw timings. Instead, it combines:

- learnable Fourier features for timing channels,
- a learned key embedding for discrete key codes,
- a learned Gaussian-style positional encoding parameterized by `mu` and `sigma`,
- a Transformer encoder stack,
- a projection head that maps the flattened sequence representation into a fixed-dimensional embedding space.

This is a meaningful design choice. It reflects an attempt to model both periodic or range-sensitive timing structure and key identity, rather than treating the sequence as only a set of scalar latencies.

The Aalto-specific training flow in [`experiments/keystroke/AaltoDB/train.py`](experiments/keystroke/AaltoDB/train.py) is especially informative because it shows the current “best” repository story. It constructs compact keystroke features, builds cached pickles, derives feature ranges for the Fourier feature module, and uses the shared Lightning trainer for embedding learning and periodic metric logging. The HMOGDB and HuMIdb keystroke paths rely on the shared nested training utility in [`experiments/keystroke/nested_training.py`](experiments/keystroke/nested_training.py), which standardizes loading, scaling, validation truncation, and metric computation for nested user/session/sequence data layouts.

### 3.3 Multimodal keystroke+IMU path

The repository also contains a large multimodal branch under [`experiments/keystroke_imu_combined`](experiments/keystroke_imu_combined). The core implementation is in [`experiments/keystroke_imu_combined/combined_training.py`](experiments/keystroke_imu_combined/combined_training.py), which defines a manual training loop and supports multiple IMU sensor subsets (accelerometer, gyroscope, magnetometer, and combinations).

Architecturally, the multimodal encoder family is still grounded in [`experiments/common/modeling.py`](experiments/common/modeling.py). The combined transformer blocks use:

- multi-head attention over the feature axis,
- separate attention over the sequence axis,
- layer normalization,
- a CNN refinement stage,
- separate stream encoders for keystrokes and IMU inputs,
- late fusion through concatenation and linear projection.

This multimodal path is an important asset for the repository because it gives the project a broader behavioral-authentication scope than keystroke-only baselines. However, it is currently maintained as a more manual and less modernized path than the Lightning-based keystroke system.

### 3.4 Data preparation and dataset handling

The repository supports AaltoDB, HMOGDB, and HuMIdb. Data preparation scripts live under [`data/AaltoDB`](data/AaltoDB), [`data/HMOGDB`](data/HMOGDB), and [`data/HuMIdb`](data/HuMIdb). The data model differs across datasets:

- AaltoDB is handled largely as user/session keystroke tables and then converted into feature arrays.
- HMOGDB and HuMIdb are stored as nested user/session/sequence structures, including IMU streams where relevant.

The keystroke training utilities apply dataset-specific scaling rules, session filtering, and validation truncation. The shared dataset helpers in [`experiments/common/datasets.py`](experiments/common/datasets.py) implement a grouped training sampler that guarantees multiple samples per user in a batch, which is a practical requirement for contrastive or metric-learning losses.

### 3.5 Evaluation protocol and metrics

The evaluation code in [`evaluation/metrics.py`](evaluation/metrics.py) is one of the repository’s stronger research assets. It provides:

- EER computation,
- distance-based verification scoring,
- usability,
- TCR,
- FAWI,
- FRWI,
- DET curve export,
- t-SNE/PCA-style visualization support.

This matters because the repository already acknowledges that CA should not be judged only by a single static operating point. The Aalto and nested keystroke training scripts also expose periodic metric functions that summarize EER, usability, TCR, FAWI, and FRWI over validation and sampled training subsets. In other words, the repository does contain a richer CA measurement vocabulary than many purely embedding-oriented biometric projects.

That said, the current metrics are still downstream analyses of embedding-based verification. The training objective itself does not yet directly optimize online detection delay or false alarm behavior.

## 4. Implementation Status and Code Reality

The current repository has real research value, but it also has real documentation and maintenance problems that should be stated directly.

First, the public-facing [`README.md`](README.md) is not actually a project overview. It currently reads as a specification for a future `TCRAwareLoss` module and a time-optimal continuous-authentication objective. That is intellectually useful, but it does not describe the implemented repository. Any paper based on this code would need documentation that aligns with the actual training and evaluation flows.

Second, the main keystroke path is the most modernized and coherent branch. The Lightning module in [`experiments/common/lightning.py`](experiments/common/lightning.py) uses supervised contrastive training, logging, model compilation, checkpointing, and the optional Muon-hybrid optimizer path. If one were choosing a single branch to anchor a paper prototype today, this would be the most defensible place to start.

Third, the multimodal and transfer-learning paths are older and less unified. The combined IMU branch still uses a separate manual loop in [`experiments/keystroke_imu_combined/combined_training.py`](experiments/keystroke_imu_combined/combined_training.py). The transfer-learning scripts define their own model and `TripletLoss` logic rather than reusing the shared common modules. This is not fatal, but it does mean the repository currently contains multiple partially overlapping training idioms.

Fourth, some branches show code drift that would need cleanup before publication-grade reproducibility claims. Two examples are especially notable:

- [`experiments/keystroke_imu_combined/combined_training.py`](experiments/keystroke_imu_combined/combined_training.py) imports `TripletLoss` from [`experiments/common/loss.py`](experiments/common/loss.py), but the shared loss module currently exposes function factories such as `triplet_loss()` and `supcon_loss()` rather than a `TripletLoss` class.
- [`experiments/keystroke/HuMIdb/model.py`](experiments/keystroke/HuMIdb/model.py) imports `TransformerEncoderLayer` from [`experiments/common/modeling.py`](experiments/common/modeling.py), but the common modeling file no longer defines that symbol in the expected legacy form.

These are precisely the kinds of issues that do not invalidate the research direction, but do weaken claims about a clean, reproducible, end-to-end artifact.

Finally, the repository now includes a meaningful optimizer experimentation hook. The shared Lightning path supports a Muon-hybrid configuration documented in [`docs/training.md`](docs/training.md) and implemented in [`experiments/common/lightning.py`](experiments/common/lightning.py). This is not the main paper contribution, but it is a genuine current capability that belongs in a faithful system description.

## 5. Methodological Gaps to Publication

The current repository is a credible research prototype, but not yet a publication-ready A*-level CA paper. The main gaps are methodological rather than cosmetic.

### 5.1 No unified paper narrative

There is currently no manuscript that matches the implemented system. The repository contains architecture, training, and evaluation code, but no coherent narrative connecting the present embedding-based design to a strong contribution claim. This draft is itself an attempt to start filling that gap.

### 5.2 No bundled benchmark table

The code computes useful metrics, but the repository does not currently bundle a stable benchmark table with dataset-by-dataset results, training settings, seeds, runtime budgets, or variance estimates. Without that, it is difficult to make strong empirical claims even if the code paths are present.

### 5.3 No explicit attack-start-aware online training

This is the largest conceptual gap. The current training code optimizes embeddings through supervised contrastive or related metric-learning objectives. Yet the `README.md` and broader CA framing point toward a much stronger claim: explicit optimization of delayed impostor detection and genuine-user usability. That capability is not yet integrated into the active training system.

### 5.4 Threshold selection realism is underdeveloped

The repository computes EER and downstream usability/security metrics, but a stronger paper would need a more explicit calibration story: where thresholds are selected, whether per-user thresholds are allowed, how operating points transfer across sessions or datasets, and whether the evaluation leaks information by tuning too close to the test distribution.

### 5.5 Stronger baselines and ablations are needed

A convincing A*-level paper would require careful comparisons against strong recent baselines such as BehaveFormer, TypeFormer, timing-aware systems such as TKCA, and possibly user-agnostic/few-shot formulations such as AuthentiSense. It would also need ablations on modality, feature design, optimizer, sequence length, enrollment budget, and thresholding strategy.

### 5.6 Reproducibility and code-path consistency need tightening

The coexistence of modernized and legacy branches, plus the examples of code drift noted above, means the repository would need cleanup, harmonization, and verification before it could serve as a polished companion artifact for a top venue.

## 6. Roadmap to a Stronger Paper

The most promising future direction is not merely “another transformer for keystrokes.” The stronger direction is a hybrid continuous-authentication formulation that keeps the repository’s useful embedding-based verification machinery while adding an explicit online risk branch.

### 6.1 Dual-output model

The model should preserve sequence embeddings for enrollment/verification while also producing per-step internal features or risk scores. This would let the system keep compatibility with biometric verification metrics while gaining an explicit online decision signal.

### 6.2 TCR-aware objective

The `README.md` already sketches the right idea: a `TCRAwareLoss` that penalizes delayed impostor detection, false alarms on genuine users, and optionally unstable temporal behavior. A strong next step would be to integrate that objective into the shared keystroke training path rather than leaving it as a standalone specification.

### 6.3 Synthetic takeover episodes

Because the current datasets do not provide explicit attack-start annotations, the repository would need a principled synthetic episode protocol. One plausible direction is to construct mixed-window takeover sequences with a genuine prefix and impostor suffix, then optimize detection relative to an injected attack start.

### 6.4 Calibrated online evaluation

A stronger paper would introduce a fixed online evaluation protocol with calibrated thresholds, explicit attack onset, time-to-detect reporting, and matched usability/security tradeoff analysis. This is more consistent with the CA threat model than relying only on session-level embedding verification.

### 6.5 Stronger baselines, ablations, and cross-dataset analysis

The future paper should compare:

- keystroke-only versus keystroke+IMU,
- static embedding loss versus TCR-aware hybrid loss,
- different enrollment budgets,
- different sequence lengths,
- different thresholding schemes,
- within-dataset and cross-dataset behavior.

That would transform the repository from a solid prototype into a paper with a more defensible and timely research claim.

## 7. Limitations

This repository is stronger as research code than as a submission-ready artifact. Its keystroke path is substantially more polished than several of its legacy branches. Not all experiment families are equally modernized, equally unified, or equally easy to reproduce. The current evaluation code is useful and richer than EER alone, but the absence of a clean bundled benchmark suite and the lack of attack-start-aware training prevent strong claims about time-optimal CA.

Accordingly, any paper-grade claims should be limited to what the repository actually implements and validates. The current code supports embedding-based continuous-authentication experiments with multi-dataset evaluation and a meaningful multimodal extension. It does not yet support strong claims about directly optimizing online detection delay. That future direction remains plausible and well motivated, but still provisional until it is implemented, benchmarked, and stress-tested.

## 8. Conclusion

The repository already contains a meaningful foundation for continuous-authentication research. It supports keystroke-only and keystroke+IMU experiments, spans three public datasets, uses transformer-family sequence models, and evaluates more than just EER. Those are real strengths. At the same time, the codebase currently tells two different stories: the implemented story is embedding-based verification with richer CA metrics, while the documented aspirational story is time-aware online authentication with a TCR-aware objective.

That split should not be ignored. Instead, it should guide the next phase of the project. The most publishable path forward is to treat the current repository as the foundation for a stronger hybrid CA system that preserves embedding-based biometric verification while adding explicit online risk scoring, synthetic takeover episodes, calibrated evaluation, and stronger comparative experiments. Until that work is completed and the codebase is tightened for reproducibility, the repository should be viewed as a strong research prototype rather than a finished A*-ready submission.

## References and Literature Anchors

- BehaveFormer: A Framework with Spatio-Temporal Dual Attention Transformers for IMU-enhanced Keystroke Dynamics. [OpenReview / IJCB 2023](https://openreview.net/forum?id=Neh0qm0MA0)
- TypeFormer: transformers for mobile keystroke biometrics. [Neural Computing and Applications, 2024](https://link.springer.com/article/10.1007/s00521-024-10140-2)
- TKCA: a timely keystroke-based continuous user authentication with short keystroke sequence in uncontrolled settings. [Cybersecurity, 2021](https://link.springer.com/article/10.1186/s42400-021-00075-9)
- AuthentiSense: A Scalable Behavioral Biometrics Authentication Scheme using Few-Shot Learning for Mobile Platforms. [NDSS 2023](https://www.ndss-symposium.org/ndss-paper/authentisense-a-scalable-behavioral-biometrics-authentication-scheme-using-few-shot-learning-for-mobile-platforms/)
- Behavioral biometrics & continuous user authentication on mobile devices: A survey. [Information Fusion, 2021](https://www.sciencedirect.com/science/article/pii/S1566253520303493)
