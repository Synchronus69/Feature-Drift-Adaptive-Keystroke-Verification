import numpy as np
from sklearn.metrics import roc_curve
 
 
def build_scaled_manhattan_profile(enroll_data):
    """
    Build a user's biometric profile (a "template") from their
    enrollment typing data.
 
    The profile is two vectors, one per keystroke feature:
      - profile_median: the user's typical value for that feature
      - profile_mad: how much that feature normally varies for this
        user (their Median Absolute Deviation — the median of how far
        each enrollment sample sits from the median)
 
    Median/MAD are used instead of mean/standard deviation because
    keystroke timing data has occasional extreme outliers (a long
    pause mid-word), and median-based statistics aren't dragged
    around by a handful of extreme values the way mean/std are.
 
    enroll_data: numpy array, shape (n_enrollment_samples, n_features)
 
    Returns: (profile_median, profile_mad), each shape (n_features,)
    """
    profile_median = np.median(enroll_data, axis=0)
    profile_mad = np.median(np.abs(enroll_data - profile_median), axis=0)
    # Guard against a feature that never varies during enrollment
    # (MAD of exactly 0 would cause a divide-by-zero later).
    profile_mad = np.where(profile_mad == 0, 1e-6, profile_mad)
    return profile_median, profile_mad
 
 
def scaled_manhattan_batch(X, profile_median, profile_mad):
    """
    Score a batch of typing attempts against a user's profile.
 
    For each attempt, this is the sum, across all features, of:
        |attempt's value for this feature - profile's median for it|
        -------------------------------------------------------------
                    profile's MAD for this feature
 
    Dividing by MAD scales each feature by how much it normally varies
    for this specific user, so a feature that's naturally noisy for
    them doesn't dominate the score, and a feature that's naturally
    very stable for them counts more heavily when it's off.
 
    A LOW score means the attempt looks like the profile owner.
    A HIGH score means it looks like someone else (or a drifted
    version of the same person — the adaptive logic elsewhere handles
    telling those two cases apart over time).
 
    X: numpy array, shape (n_attempts, n_features)
    profile_median, profile_mad: from build_scaled_manhattan_profile
 
    Returns: numpy array of scores, shape (n_attempts,)
    """
    return np.sum(np.abs(X - profile_median) / profile_mad, axis=1)
 
 
def find_balanced_accuracy_threshold(y_true, y_score):
    """
    Pick a decision threshold using balanced accuracy rather than raw
    accuracy.
 
    Raw accuracy is a trap here: genuine attempts are usually far
    outnumbered by impostor attempts in an evaluation set, so a
    threshold that just rejects almost everything can post a high raw
    accuracy while actually failing real users constantly (high FRR).
    Balanced accuracy is the average of the true positive rate
    (correctly accepting the real user) and the true negative rate
    (correctly rejecting impostors), so both kinds of mistake count
    equally regardless of how many of each type are in the data.
 
    y_true: 1 = impostor attempt, 0 = genuine attempt (this matches
            how the score is defined: higher score = more impostor-like)
    y_score: the scaled-Manhattan (or other) distance score per attempt
 
    Returns: the threshold that maximizes balanced accuracy
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    tnr = 1 - fpr
    balanced_acc = (tpr + tnr) / 2
    best_idx = np.argmax(balanced_acc)
    return thresholds[best_idx]
 
 
def threshold_at_far(y_true, y_score, target_far):
    """
    Find the score threshold that would produce a specific target
    false-acceptance rate (FAR) if used as the decision boundary.
 
    This is used for the "adapt" threshold in the two-threshold gate,
    not the strict access threshold. The idea: rather than picking an
    arbitrary looser margin (e.g. "1.5x the strict threshold") to
    decide what the profile is allowed to learn from, anchor it to a
    named, interpretable false-acceptance rate instead — "we're
    willing to let in impostor-like attempts up to X% of the time, in
    exchange for the profile being able to track real behavioral
    drift."
 
    y_true: 1 = impostor attempt, 0 = genuine attempt
    y_score: the scaled-Manhattan (or other) distance score per attempt
    target_far: desired false-acceptance rate, e.g. 0.05 for 5%
 
    Returns: the score threshold whose FAR is closest to target_far
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(fpr - target_far))
    return thresholds[idx]