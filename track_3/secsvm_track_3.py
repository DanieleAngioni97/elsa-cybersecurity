import json
import os
from src.models import SecSVM
from src.models.utils import *


if __name__ == "__main__":
    """
    NB: in this example, the pre-extracted features are used. Alternatively,
    the APK file paths can be passed to the classifier.
    To fit the model, you can use `classifier.extract_features` to get the
    features and then pass them to `classifier.fit`.
    To classify the APK files, you can directly pass the list containing the
    file paths to `classifier.classify`.
    """
    classifier = SecSVM(C=0.1, lb=-0.5, ub=0.5)

    base_path = os.path.dirname(__file__)

    clf_path = os.path.join(
        base_path, "../pretrained/secsvm_classifier.pkl")
    vect_path = os.path.join(
        base_path, "../pretrained/secsvm_vectorizer.pkl")

    if os.path.exists(clf_path) and os.path.exists(vect_path):
        classifier = SecSVM.load(vect_path, clf_path)
    else:
        features_tr = load_features(
            os.path.join(base_path, "data/training_set_features.zip"))
        y_tr = load_labels(
            os.path.join(base_path, "data/training_set_features.zip"),
            os.path.join(base_path, "data/training_set.zip"))
        classifier.fit(features_tr, y_tr)
        classifier.save(vect_path, clf_path)

    results = []
    for i in range(1, 5):
        features_ts = load_features(
            os.path.join(base_path, f"data/test_set_features_round_{i}.zip"))
        y_pred, scores = classifier.predict(features_ts)
        results.append({
            sha256: [int(y), float(s)] for sha256, y, s in zip(
                load_sha256_list(os.path.join(
                    base_path, f"data/test_set_features_round_{i}.zip")),
                y_pred, scores)})

    with open(os.path.join(base_path,
                           "submissions/submission_secsvm_track_3.json"),
              "w") as f:
        json.dump(results, f)
