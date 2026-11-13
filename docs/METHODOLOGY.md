# Research methodology

**Data.** MIT-BIH Arrhythmia Database, 48 records, 360 Hz, 2 leads (typically MLII + V1/V2/V4/V5). Beat annotations mapped to AAMI classes
(N: N,L,R,e,j; S: A,a,J,S; V: V,E; F: F; Q: /,f,Q). Non-beat symbols are excluded (listed in `config.EXCLUDED_SYMBOLS`).

**Preprocessing.** Artefact clipping -> two-stage median baseline estimate (200 ms, 600 ms) subtracted -> optional IIR notch -> 4th-order zero-phase
Butterworth band-pass 0.5-40 Hz -> record-level scaling by median |R amplitude| -> beat window [-250 ms, +400 ms].
For training, annotated R locations are refined to the local maximum absolute amplitude (+/-50 ms). The detector used when no annotations are supplied is Pan-Tompkins-style: 5-15 Hz band-pass, derivative, squaring, 120 ms moving integration, a global threshold of 30% of the 99.5th percentile of the integrated signal, and a 250 ms refractory period. Candidate positions are refined to the local maximum absolute amplitude.

**Features.** RR: previous, next, local mean/std (10 intervals), ratios, heart rate. Per channel: R amplitude, QRS duration (width of region with
|slope| > 20% of max), energy, variance, skewness, kurtosis, peak-to-peak, area, pre/post-R means, T-wave amplitude, dominant frequency,
PSD fraction in 0.5-5 / 5-15 / 15-40 Hz, spectral entropy, relative db4 level-4 wavelet energies (A4, D4, D3, D2).

**Rough sets.** On a class-balanced train sample, each feature is discretised into 4 equal-frequency bins (cuts from train). For attribute subset B,
IND(B) partitions objects; the lower approximation of class X is the union of equivalence classes fully inside X, the upper approximation those intersecting X;
POS_B(d) is the union of lower approximations of all decision classes; gamma_B(d) = |POS_B(d)|/|U|. Attribute significance = gamma(B+{a}) - gamma(B).
The reduct is built greedily (QuickReduct) until gamma(R) = gamma(C) (tolerance configurable) and pruned by backward elimination. Core = attributes whose removal lowers gamma(C).
Limitation: with many features equal-frequency bins make most equivalence classes small, so gamma(C) approaches 1 and the reduct is a consistency-preserving, not accuracy-optimised, subset.

**Fuzzy classifier.** Each feature has Low/Medium/High terms (left-shoulder, triangular and right-shoulder memberships meeting at training quantiles 15/50/85%). Membership quantiles are fitted on a deterministic class-balanced training sample only; constant features receive a small non-zero shoulder span. Rules are conjunctions of up to 3 distinct features. For each class, a beam search scores antecedents by class-normalised fuzzy support and fuzzy confidence. A confidence-weighted marginal-coverage step selects up to 12 rules/class; rule weight = confidence. Firing = product t-norm; class score = 1 - prod(1 - weight*firing); row-normalised scores are used for comparison and prediction = argmax. These scores are not calibrated probabilities. Explanations include feature memberships, fired rules, top-two score margin, active-rule count and "no rule fired" status.

**Experiments.** Non-finite extracted feature values are replaced with training-set medians; the same training medians are applied to test rows. Models A (fuzzy, all features), B (rough set + fuzzy), C1 (Random Forest, 200 trees), C2 (RBF-SVM) share the same record-wise partitions.
Metrics: accuracy, balanced accuracy, per-class precision / recall(sensitivity) / specificity / F1, weighted F1, macro-F1 over classes represented in the test set, macro-F1 over all five AAMI classes, confusion matrices, one-vs-rest ROC-AUC and average precision where both positive and negative examples are present, and train/predict time. Missing classes are retained in the fixed-order confusion matrix and explicitly reported in the run manifest.
Statistical caveat: a single DS1/DS2 split gives no confidence intervals; repeat with other record partitions for stronger claims.
