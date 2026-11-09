"""Convert the Kaggle/PhysioNet PTB-XL release to this repo's .npy layout."""

import argparse
import ast
import csv
import os
from collections import defaultdict

import numpy as np
import wfdb


STTC_CODES = {
    "NDT", "NST_", "DIG", "ISC_", "ISCAL", "LNGQT", "ISCIN", "ISCIL",
    "ISCAS", "ISCLA", "ANEUR", "EL", "ISCAN",
}
MI_CODES = {
    "IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI", "INJAL",
    "IPLMI", "IPMI", "INJIN", "INJLA", "PMI", "INJIL",
}
HYP_CODES = {"LVH", "LAO/LAE", "RVH", "RAO/RAE", "SEHYP"}
CD_CODES = {
    "LAFB", "IRBBB", "1AVB", "IVCD", "CRBBB", "CLBBB", "LPFB", "WPW",
    "ILBBB", "3AVB", "2AVB",
}


def diagnosis_group(code):
    if code == "NORM":
        return "NORM"
    if code in MI_CODES:
        return "MI"
    if code in STTC_CODES:
        return "STTC"
    if code in CD_CODES:
        return "CD"
    if code in HYP_CODES:
        return "HYP"
    return "others"


def final_diagnosis(raw_codes):
    codes = ast.literal_eval(raw_codes)
    if not codes:
        return "others"
    # Retain the first code in the event of a probability tie, as in the notebook.
    top_code = max(codes, key=codes.get)
    return diagnosis_group(top_code)


def resample_and_normalize(signal, sampling_rate=500):
    target_len = int(signal.shape[0] / 500 * sampling_rate)
    old_t = np.linspace(1, signal.shape[0], signal.shape[0])
    new_t = np.linspace(1, signal.shape[0], target_len)
    resampled = np.stack(
        [np.interp(new_t, old_t, signal[:, ch]) for ch in range(signal.shape[1])],
        axis=1,
    )
    mean = resampled.mean(axis=0, keepdims=True)
    std = resampled.std(axis=0, keepdims=True)
    std[std == 0] = 1
    return ((resampled - mean) / std).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="Extracted PTB-XL release directory")
    parser.add_argument("--output-root", required=True, help="Output dataset root for this repo")
    parser.add_argument("--sampling-rate", type=int, default=500, choices=(250, 500), help="Output sampling rate in Hz")
    args = parser.parse_args()

    raw_root = os.path.abspath(args.raw_root)
    output_root = os.path.abspath(args.output_root)
    metadata_path = os.path.join(raw_root, "ptbxl_database.csv")
    feature_dir = os.path.join(output_root, "Feature")
    label_dir = os.path.join(output_root, "Label")
    os.makedirs(feature_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    patients = defaultdict(list)
    with open(metadata_path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            group = final_diagnosis(row["scp_codes"])
            patients[int(float(row["patient_id"]))].append((group, row["filename_hr"]))

    label_ids = []
    for patient_id in sorted(patients):
        records = patients[patient_id]
        groups = {group for group, _ in records}
        if len(groups) == 1 and "others" not in groups:
            label_ids.append((patient_id, groups.pop()))

    class_ids = {"NORM": 0, "MI": 1, "STTC": 2, "CD": 3, "HYP": 4}
    labels = []
    total_segments = 0
    for subject_number, (patient_id, group) in enumerate(label_ids, start=1):
        trials = []
        for _, relative_path in patients[patient_id]:
            signal, _ = wfdb.rdsamp(os.path.join(raw_root, relative_path))
            normalized = resample_and_normalize(signal, args.sampling_rate)
            # Split each record into non-overlapping one-second segments.
            usable = (normalized.shape[0] // args.sampling_rate) * args.sampling_rate
            trials.append(normalized[:usable].reshape(-1, args.sampling_rate, normalized.shape[1]))

        subject_features = np.concatenate(trials, axis=0)
        np.save(os.path.join(feature_dir, f"feature_{subject_number:05d}.npy"), subject_features)
        labels.append((class_ids[group], subject_number))
        total_segments += len(subject_features)
        if subject_number % 500 == 0:
            print(f"Processed {subject_number}/{len(label_ids)} subjects ({total_segments} segments)", flush=True)

    np.save(os.path.join(label_dir, "label.npy"), np.asarray(labels, dtype=np.int64))
    print(
        f"Complete: {len(labels)} subjects, {total_segments} segments, "
        f"classes={dict((name, sum(y == class_id for y, _ in labels)) for name, class_id in class_ids.items())}"
    )


if __name__ == "__main__":
    main()
