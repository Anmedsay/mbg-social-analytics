
#!/usr/bin/env python3
"""
TK3 Analitika Media Sosial - Analisis Sentimen RQ3
Topik: Keamanan dan kebersihan pangan MBG di YouTube dan X.

Input:
  mbg_processed.csv, atau ZIP yang memuat mbg_processed.csv.

Output:
  Prediksi sentimen, audit data, tabel, grafik, sampel anotasi,
  evaluasi precision/recall/F1, dan log terminal.

Model:
  w11wo/indonesian-roberta-base-sentiment-classifier

Cara menggunakan:
  python3 tk3_sentimen_rq3.py --mode run
  python3 tk3_sentimen_rq3.py --mode report
  python3 tk3_sentimen_rq3.py --mode report --topics topik_rq1.csv
"""

import argparse
import csv
import hashlib
import json
import platform as py_platform
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

MODEL_ID = "w11wo/indonesian-roberta-base-sentiment-classifier"
LABELS = ["negative", "neutral", "positive"]
SEED = 42
random.seed(SEED)

COLUMNS = {
    "doc_id", "platform", "doc_type", "rq3_eligible",
    "rq3_label_sample", "rq3_text", "rq1_issue_categories",
    "common_year_month", "common_in_overlap_period",
    "common_is_spam", "is_relevant_candidate",
}

PRED_COLUMNS = [
    "doc_id", "platform", "doc_type", "month",
    "in_overlap", "issue_keywords", "sentiment",
    "confidence", "excerpt",
]

OUT = None


def log(message=""):
    """Cetak ke terminal sekaligus simpan bukti proses."""
    message = str(message)
    print(message, flush=True)
    if OUT is not None:
        with (OUT / "terminal_ringkasan.txt").open(
            "a", encoding="utf-8"
        ) as file:
            file.write(message + "\n")


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def choose_file():
    """Pilih file dari komputer, tanpa mengunggahnya ke server."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Pilih mbg_processed.csv atau ZIP dataset TK2",
            filetypes=[
                ("Dataset CSV / ZIP", "*.csv *.zip"),
                ("Semua file", "*.*"),
            ],
        )
        root.destroy()
        if selected:
            return Path(selected)
    except Exception:
        pass

    return Path(input("Masukkan path dataset CSV/ZIP: ").strip().strip('"'))


def scrub(text, limit=None):
    """Samarkan sebagian informasi identitas dalam teks."""
    text = str(text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text
    )
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(
        r"(?<!\d)(?:\+62|62|08)[\d\s-]{8,15}",
        "[NOMOR]", text
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] if limit else text


def read_dataset(path, chunk_size):
    """Baca CSV/ZIP bertahap agar penggunaan RAM terkendali."""
    options = {
        "dtype": str,
        "keep_default_na": False,
        "chunksize": chunk_size,
        "encoding": "utf-8",
        "usecols": lambda column: column in COLUMNS,
    }

    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            names = [
                name for name in archive.namelist()
                if Path(name).name.lower() == "mbg_processed.csv"
                and not name.startswith("__MACOSX/")
            ]
            if not names:
                raise ValueError(
                    "ZIP tidak memuat mbg_processed.csv. "
                    "Pilih hasil akhir TK2, bukan dataset mentah."
                )
            if len(names) > 1:
                raise ValueError(
                    "Ada lebih dari satu mbg_processed.csv dalam ZIP."
                )
            log(f"CSV dalam ZIP: {names[0]}")
            with archive.open(names[0]) as file:
                yield from pd.read_csv(file, **options)
    else:
        if path.suffix.lower() != ".csv":
            raise ValueError("File harus berformat CSV atau ZIP.")
        yield from pd.read_csv(path, **options)


def data_signature(path, args):
    stat = path.stat()
    return {
        "file_name": path.name,
        "file_size": stat.st_size,
        "file_mtime_ns": stat.st_mtime_ns,
        "model": args.model,
        "max_length": args.max_length,
        "seed": SEED,
    }


def choose_device(requested):
    if requested != "auto":
        return requested

    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def normal_label(value):
    mapping = {
        "negatif": "negative",
        "negative": "negative",
        "netral": "neutral",
        "neutral": "neutral",
        "positif": "positive",
        "positive": "positive",
    }
    return mapping.get(str(value).lower().strip(), "")


def write_annotation_template(samples):
    file = OUT / "anotasi_manual_800.csv"

    if file.exists():
        log("File anotasi manual sudah ada; tidak ditimpa.")
        return

    rows = []
    for item in sorted(samples, key=lambda x: (x["platform"], x["doc_id"])):
        rows.append({
            "doc_id": item["doc_id"],
            "platform": item["platform"],
            "text_for_annotation": scrub(item["rq3_text"]),
            "annotator_1": "",
            "annotator_2": "",
            "gold_label": "",
            "notes": "",
        })

    pd.DataFrame(rows).to_csv(
        file, index=False, encoding="utf-8-sig"
    )
    log(f"Sampel anotasi manual: {len(rows):,} dokumen")
    log(f"File: {file.name}")


def predict(args):
    from transformers import pipeline
    import torch
    import transformers

    path = Path(args.input) if args.input else choose_file()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    log("=" * 62)
    log("TK3 | ANALISIS SENTIMEN MBG | RQ3")
    log("=" * 62)
    log(f"Dataset: {path.name}")
    log(f"Ukuran: {path.stat().st_size / 1e6:,.1f} MB")
    log(f"Python: {sys.version.split()[0]}")
    log(f"Pandas: {pd.__version__}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Transformers: {transformers.__version__}")

    signature = data_signature(path, args)
    cfg_path = OUT / "konfigurasi_run.json"
    pred_path = OUT / "prediksi_rq3.csv"

    if pred_path.exists():
        if not cfg_path.exists():
            raise ValueError(
                "Ada file prediksi tetapi konfigurasi hilang. "
                "Jangan mencampurkan hasil dua eksekusi."
            )
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old["input_signature"] != signature:
            raise ValueError(
                "Dataset atau pengaturan model berubah. "
                "Gunakan --output folder_baru untuk run baru."
            )
        existing = set(
            pd.read_csv(
                pred_path, usecols=["doc_id"], dtype=str
            )["doc_id"]
        )
        log(f"Mode resume: {len(existing):,} prediksi sudah tersimpan")
    else:
        existing = set()

    device = choose_device(args.device)
    log(f"Model: {args.model}")
    log(f"Perangkat inferensi: {device}")
    log(f"Batch size: {args.batch_size}")
    log(f"Panjang maksimum token: {args.max_length}")

    classifier = pipeline(
        "text-classification",
        model=args.model,
        tokenizer=args.model,
        device=device,
    )

    labels_model = {
        normal_label(label)
        for label in classifier.model.config.id2label.values()
    }
    if labels_model != set(LABELS):
        raise ValueError(
            "Label model tidak sesuai dengan tiga kelas RQ3: "
            f"{classifier.model.config.id2label}"
        )

    commit = getattr(
        classifier.model.config, "_commit_hash", None
    )
    if not cfg_path.exists():
        cfg_path.write_text(
            json.dumps({
                "input_signature": signature,
                "model_revision": commit,
                "device": device,
                "python": sys.version,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    log(f"Model revision: {commit or 'tidak tersedia'}")
    log("Memulai audit dan klasifikasi dataset...")

    total_rows = 0
    total_eligible = 0
    per_platform = Counter()
    all_eligible_ids = set()
    flagged_samples = []
    sample_reservoir = {"youtube": [], "x": []}
    reservoir_seen = Counter()
    new_predictions = 0

    with pred_path.open(
        "a", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=PRED_COLUMNS)
        if file.tell() == 0:
            writer.writeheader()

        for chunk_no, chunk in enumerate(
            read_dataset(path, args.chunk_size), 1
        ):
            missing = {
                "doc_id", "platform", "doc_type",
                "rq3_eligible", "rq3_text",
            } - set(chunk.columns)

            if missing:
                raise ValueError(
                    f"Kolom penting tidak ada: {sorted(missing)}"
                )

            total_rows += len(chunk)
            part = chunk[
                chunk["rq3_eligible"].map(is_true)
            ].copy()

            total_eligible += len(part)
            per_platform.update(part["platform"].value_counts().to_dict())

            if "common_is_spam" in part:
                if part["common_is_spam"].map(is_true).any():
                    raise ValueError(
                        "Dokumen spam masih lolos ke RQ3."
                    )
            if (part["doc_type"] == "video").any():
                raise ValueError(
                    "Video masuk RQ3, padahal seharusnya opini pengguna."
                )
            if part["rq3_text"].str.strip().eq("").any():
                raise ValueError(
                    "Ada dokumen RQ3 dengan teks kosong."
                )

            records = part.to_dict("records")

            for row in records:
                doc_id = row["doc_id"]
                if doc_id in all_eligible_ids:
                    raise ValueError(
                        f"Duplikasi doc_id terpilih: {doc_id}"
                    )
                all_eligible_ids.add(doc_id)

                platform_name = row["platform"]
                if is_true(row.get("rq3_label_sample", "")):
                    flagged_samples.append(row)

                if platform_name in sample_reservoir:
                    reservoir_seen[platform_name] += 1
                    n_seen = reservoir_seen[platform_name]
                    reservoir = sample_reservoir[platform_name]

                    if len(reservoir) < 400:
                        reservoir.append(row)
                    else:
                        j = random.randrange(n_seen)
                        if j < 400:
                            reservoir[j] = row

            pending = [
                row for row in records
                if row["doc_id"] not in existing
            ]

            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                texts = [r["rq3_text"] for r in batch]

                predictions = classifier(
                    texts,
                    truncation=True,
                    max_length=args.max_length,
                    batch_size=args.batch_size,
                )

                for row, result in zip(batch, predictions):
                    label = normal_label(result["label"])
                    if label not in LABELS:
                        raise ValueError(
                            f"Label model tidak dikenal: {result['label']}"
                        )

                    writer.writerow({
                        "doc_id": row["doc_id"],
                        "platform": row["platform"],
                        "doc_type": row["doc_type"],
                        "month": row.get("common_year_month", ""),
                        "in_overlap": row.get(
                            "common_in_overlap_period", ""
                        ),
                        "issue_keywords": row.get(
                            "rq1_issue_categories", ""
                        ),
                        "sentiment": label,
                        "confidence": round(float(result["score"]), 6),
                        "excerpt": scrub(row["rq3_text"], 200),
                    })
                    new_predictions += 1

            file.flush()
            log(
                f"Chunk {chunk_no:>4}: "
                f"{total_rows:,} baris dibaca | "
                f"{total_eligible:,} layak RQ3 | "
                f"{new_predictions:,} prediksi baru"
            )

    sample = flagged_samples
    if not sample:
        sample = (
            sample_reservoir["youtube"]
            + sample_reservoir["x"]
        )
        log(
            "Flag sampel TK2 tidak ditemukan; "
            "digunakan reservoir sampling per platform."
        )

    write_annotation_template(sample)

    audit = {
        "rows_read": total_rows,
        "rq3_eligible": total_eligible,
        "rq3_by_platform": dict(per_platform),
        "annotation_sample": len(sample),
        "new_predictions": new_predictions,
        "source_file": path.name,
    }

    (OUT / "audit_dataset.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("\n" + "=" * 62)
    log("AUDIT DATASET SELESAI")
    log("=" * 62)
    log(f"Total baris sumber: {total_rows:,}")
    log(f"Total dokumen RQ3 tingkat A: {total_eligible:,}")
    for name in ["youtube", "x"]:
        log(f"  {name}: {per_platform[name]:,}")
    log(f"Sampel untuk anotasi manual: {len(sample):,}")

    report(args)


def plot_and_save(fig, filename):
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_distribution(pred):
    count = (
        pred["sentiment"]
        .value_counts()
        .reindex(LABELS, fill_value=0)
    )
    share = count / len(pred) * 100

    table = pd.DataFrame({
        "sentiment": LABELS,
        "jumlah": count.values,
        "persentase": share.round(3).values,
    })
    table.to_csv(
        OUT / "distribusi_sentimen.csv", index=False
    )

    log("\nDISTRIBUSI SENTIMEN KESELURUHAN")
    log(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(table["sentiment"], table["persentase"])
    ax.set_title("Distribusi Sentimen RQ3")
    ax.set_ylabel("Persentase dokumen (%)")
    ax.set_ylim(0, max(5, table["persentase"].max() * 1.18))
    plot_and_save(fig, "01_distribusi_sentimen.png")


def save_platform_comparison(pred):
    counts = pd.crosstab(
        pred["platform"], pred["sentiment"]
    ).reindex(columns=LABELS, fill_value=0)

    percentages = (
        counts.div(counts.sum(axis=1), axis=0) * 100
    ).round(3)

    counts.to_csv(OUT / "sentimen_platform_jumlah.csv")
    percentages.to_csv(
        OUT / "sentimen_platform_persentase.csv"
    )

    log("\nPERBANDINGAN SENTIMEN ANTARPLATFORM")
    log(percentages.to_string())

    fig, ax = plt.subplots(figsize=(9, 5))
    percentages.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Persentase dokumen (%)")
    ax.set_xlabel("Platform")
    ax.set_title("Komposisi Sentimen YouTube dan X")
    ax.legend(title="Sentimen")
    ax.tick_params(axis="x", rotation=0)
    plot_and_save(fig, "02_sentimen_platform.png")


def save_temporal(pred, min_docs=30):
    table = (
        pred.groupby(["platform", "month"])
        .agg(
            n=("sentiment", "size"),
            negative_n=(
                "sentiment",
                lambda x: int((x == "negative").sum())
            ),
        )
        .reset_index()
    )
    table["negative_pct"] = (
        100 * table["negative_n"] / table["n"]
    ).round(3)
    table.to_csv(
        OUT / "sentimen_per_bulan.csv", index=False
    )

    log("\nTREN SENTIMEN NEGATIF PER BULAN")
    log(
        table[
            ["platform", "month", "n", "negative_pct"]
        ].to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, subset in table.groupby("platform"):
        valid = subset[subset["n"] >= min_docs].sort_values("month")
        ax.plot(
            valid["month"], valid["negative_pct"],
            marker="o", label=name
        )

    ax.set_title("Proporsi Sentimen Negatif per Bulan")
    ax.set_xlabel("Bulan publikasi (WIB)")
    ax.set_ylabel("Sentimen negatif (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=65)
    ax.legend()
    plot_and_save(fig, "03_tren_sentimen_negatif.png")

    log(
        "PERINGATAN: tren ini menggambarkan sampel terkumpul, "
        "bukan seluruh percakapan publik."
    )


def save_topic_analysis(pred, topic_file=None):
    if topic_file:
        topics = pd.read_csv(
            topic_file, dtype=str, keep_default_na=False
        )
        if "doc_id" not in topics.columns:
            raise ValueError(
                "File topik harus mengandung kolom doc_id."
            )

        topic_col = next(
            (
                c for c in [
                    "topic_name", "topic_label", "topic"
                ] if c in topics.columns
            ), None
        )
        if topic_col is None:
            raise ValueError(
                "File topik perlu kolom topic_name, "
                "topic_label, atau topic."
            )
        if topics["doc_id"].duplicated().any():
            raise ValueError(
                "doc_id pada file topik harus unik."
            )

        joined = pred.merge(
            topics[["doc_id", topic_col]],
            on="doc_id", how="inner"
        )
        joined = joined.rename(
            columns={topic_col: "topic_label"}
        )
        joined = joined[joined["topic_label"].str.strip() != ""]

        title = "Topik hasil topic modeling RQ1"
        filename = "sentimen_per_topik_model.csv"

        log("\nANALISIS SENTIMEN LINTAS TOPIK RQ1")
        log(
            f"Dokumen yang cocok melalui doc_id: "
            f"{len(joined):,} dari {len(pred):,}"
        )
    else:
        joined = pred[
            ["doc_id", "sentiment", "issue_keywords"]
        ].copy()
        joined["topic_label"] = (
            joined["issue_keywords"].str.split(";")
        )
        joined = joined.explode("topic_label")
        joined["topic_label"] = (
            joined["topic_label"].fillna("").str.strip()
        )
        joined = joined[joined["topic_label"] != ""]

        title = "Kategori isu berbasis kata kunci (bukan topic modeling)"
        filename = "sentimen_per_kategori_kata_kunci.csv"

        log("\nANALISIS AWAL KATEGORI ISU")
        log(
            "Belum ada file topik RQ1. "
            "Kategori pada grafik ini hanya label berbasis kata kunci."
        )

    if joined.empty:
        log("Tidak tersedia kategori/topik yang dapat dibandingkan.")
        return

    counts = pd.crosstab(
        joined["topic_label"], joined["sentiment"]
    ).reindex(columns=LABELS, fill_value=0)

    counts["n"] = counts.sum(axis=1)
    percentages = (
        counts[LABELS].div(counts["n"], axis=0) * 100
    ).round(3)

    result = counts.join(
        percentages.add_suffix("_pct")
    ).sort_values("n", ascending=False)
    result.to_csv(OUT / filename)

    log(result.head(12).to_string())

    chosen = result[
        result["n"] >= 30
    ].head(10)

    if not chosen.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        chosen[
            [c + "_pct" for c in LABELS]
        ].plot(kind="barh", stacked=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Persentase dokumen (%)")
        ax.set_ylabel("Topik atau kategori")
        ax.invert_yaxis()
        plot_and_save(fig, "04_sentimen_per_topik.png")

    if not topic_file:
        log(
            "CATATAN: satu dokumen dapat masuk beberapa kategori kata kunci."
        )


def save_examples(pred):
    samples = []
    for label in LABELS:
        part = pred[pred["sentiment"] == label]
        high = part[part["confidence"] >= 0.75]

        if len(high) >= 3:
            part = high

        if not part.empty:
            sample = part.sample(
                min(3, len(part)), random_state=SEED
            )
            samples.append(sample)

    if samples:
        examples = pd.concat(samples, ignore_index=True)
        examples[
            ["doc_id", "platform", "sentiment",
             "confidence", "excerpt"]
        ].to_csv(
            OUT / "contoh_per_kelas_perlu_verifikasi.csv",
            index=False, encoding="utf-8-sig"
        )
    log(
        "\nContoh kandidat setiap kelas telah disimpan. "
        "Verifikasi manual sebelum digunakan dalam laporan."
    )


def evaluate(pred):
    annotation_path = OUT / "anotasi_manual_800.csv"

    if not annotation_path.exists():
        log("\nEVALUASI: file anotasi belum tersedia.")
        return

    labels = pd.read_csv(
        annotation_path, dtype=str, keep_default_na=False
    )
    needed = {"doc_id", "gold_label"}

    if not needed.issubset(labels.columns):
        raise ValueError(
            "File anotasi harus memuat doc_id dan gold_label."
        )
    if labels["doc_id"].duplicated().any():
        raise ValueError(
            "doc_id duplikat pada file anotasi manual."
        )

    gold_raw = labels["gold_label"].str.strip()
    annotated = labels[gold_raw != ""].copy()

    if annotated.empty:
        log(
            "\nEVALUASI BELUM DAPAT DIHITUNG: "
            "kolom gold_label belum diisi."
        )
        log(
            "Isi positif/negatif/netral secara manual, "
            "kemudian jalankan --mode report."
        )
        return

    annotated["gold"] = (
        annotated["gold_label"].map(normal_label)
    )

    if (annotated["gold"] == "").any():
        raise ValueError(
            "Ada label anotasi tidak valid. "
            "Gunakan positif, negatif, atau netral."
        )

    data = annotated.merge(
        pred[["doc_id", "sentiment"]],
        on="doc_id",
        how="inner",
        validate="one_to_one"
    )

    log("\n" + "=" * 62)
    log("EVALUASI BERDASARKAN ANOTASI MANUAL")
    log("=" * 62)
    log(f"Dokumen berlabel: {len(annotated):,}")
    log(f"Dokumen berhasil dipasangkan: {len(data):,}")

    if len(data) != len(annotated):
        raise ValueError(
            "Sebagian doc_id anotasi tidak ditemukan pada prediksi."
        )

    y_true = data["gold"]
    y_pred = data["sentiment"]

    report = classification_report(
        y_true, y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )
    table = pd.DataFrame(report).T
    table.to_csv(OUT / "evaluasi_precision_recall_f1.csv")

    matrix = confusion_matrix(
        y_true, y_pred, labels=LABELS
    )
    cm = pd.DataFrame(
        matrix,
        index=["actual_" + x for x in LABELS],
        columns=["pred_" + x for x in LABELS],
    )
    cm.to_csv(OUT / "confusion_matrix.csv")

    log(f"Akurasi: {accuracy_score(y_true, y_pred):.4f}")
    log(
        f"Macro precision: "
        f"{report['macro avg']['precision']:.4f}"
    )
    log(
        f"Macro recall: "
        f"{report['macro avg']['recall']:.4f}"
    )
    log(
        f"Macro F1-score: "
        f"{report['macro avg']['f1-score']:.4f}"
    )
    log("\nMETRIK PER KELAS")
    log(table.loc[LABELS].round(4).to_string())
    log("\nCONFUSION MATRIX")
    log(cm.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Prediksi model")
    ax.set_ylabel("Label anotasi manual")
    ax.set_title("Confusion Matrix RQ3")

    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center"
            )

    plot_and_save(fig, "05_confusion_matrix.png")

    platform_results = []
    for name, part in data.groupby("platform"):
        metrics = classification_report(
            part["gold"], part["sentiment"],
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        platform_results.append({
            "platform": name,
            "n": len(part),
            "accuracy": accuracy_score(
                part["gold"], part["sentiment"]
            ),
            "macro_precision": metrics["macro avg"]["precision"],
            "macro_recall": metrics["macro avg"]["recall"],
            "macro_f1": metrics["macro avg"]["f1-score"],
        })

    by_platform = pd.DataFrame(platform_results)
    by_platform.to_csv(
        OUT / "evaluasi_per_platform.csv",
        index=False
    )
    log("\nEVALUASI PER PLATFORM")
    log(by_platform.round(4).to_string(index=False))

    if {
        "annotator_1", "annotator_2"
    }.issubset(annotated.columns):
        pair = annotated[
            (annotated["annotator_1"].str.strip() != "")
            & (annotated["annotator_2"].str.strip() != "")
        ].copy()

        if len(pair) >= 2:
            pair["a1"] = pair["annotator_1"].map(normal_label)
            pair["a2"] = pair["annotator_2"].map(normal_label)
            pair = pair[
                (pair["a1"] != "") & (pair["a2"] != "")
            ]
            if len(pair) >= 2:
                kappa = cohen_kappa_score(
                    pair["a1"], pair["a2"],
                    labels=LABELS
                )
                log(
                    f"\nCohen's kappa dua anotator "
                    f"(n={len(pair)}): {kappa:.4f}"
                )

    mistakes = data[data["gold"] != data["sentiment"]]
    mistakes[
        ["doc_id", "platform", "gold", "sentiment"]
    ].to_csv(
        OUT / "kesalahan_klasifikasi.csv",
        index=False
    )
    log(
        f"Jumlah kesalahan klasifikasi pada sampel: "
        f"{len(mistakes):,}"
    )


def report(args):
    file = OUT / "prediksi_rq3.csv"

    if not file.exists():
        raise FileNotFoundError(
            "Belum ada prediksi. Jalankan --mode run dahulu."
        )

    pred = pd.read_csv(
        file, dtype=str, keep_default_na=False
    )

    if pred.empty:
        raise ValueError("File prediksi kosong.")
    if pred["doc_id"].duplicated().any():
        raise ValueError(
            "Ada doc_id ganda dalam hasil prediksi."
        )
    if not set(pred["sentiment"]).issubset(LABELS):
        raise ValueError(
            "Terdapat label sentimen yang tidak dikenal."
        )

    pred["confidence"] = pd.to_numeric(
        pred["confidence"], errors="raise"
    )

    log("\n" + "=" * 62)
    log("HASIL PENELITIAN | RQ3 SENTIMENT ANALYSIS")
    log("=" * 62)
    log(f"Total dokumen terklasifikasi: {len(pred):,}")
    log(
        "Jumlah per platform: "
        + str(pred["platform"].value_counts().to_dict())
    )

    save_distribution(pred)

    overlap = pred[
        pred["in_overlap"].map(is_true)
    ].copy()

    if not overlap.empty:
        log(
            f"\nPerbandingan antarplatform menggunakan "
            f"{len(overlap):,} dokumen pada periode tumpang tindih."
        )
        save_platform_comparison(overlap)
    else:
        log(
            "Penanda overlap tidak tersedia. "
            "Perbandingan memakai semua periode."
        )
        save_platform_comparison(pred)

    save_temporal(pred)
    save_topic_analysis(pred, args.topics)
    save_examples(pred)
    evaluate(pred)

    log("\n" + "=" * 62)
    log("STATUS SELESAI")
    log("=" * 62)
    log(f"Output tersimpan di: {OUT.resolve()}")
    log(
        "Catatan: confidence model bukan ukuran akurasi; "
        "precision, recall, dan F1 memerlukan gold_label."
    )
    log(
        "Hasil hanya mewakili dokumen yang terkumpul dan "
        "memenuhi kriteria, bukan seluruh masyarakat Indonesia."
    )


def main():
    global OUT

    parser = argparse.ArgumentParser(
        description="Analisis sentimen TK3 untuk RQ3 MBG."
    )
    parser.add_argument(
        "--mode", choices=["run", "report"], default="run"
    )
    parser.add_argument(
        "--input",
        help="Path CSV atau ZIP; kosong = buka pemilih file."
    )
    parser.add_argument(
        "--output", default="hasil_RQ3",
        help="Folder penyimpanan hasil."
    )
    parser.add_argument(
        "--topics",
        help="CSV topik RQ1: doc_id dan topic_name/topic_label/topic."
    )
    parser.add_argument(
        "--model", default=MODEL_ID
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16
    )
    parser.add_argument(
        "--chunk-size", type=int, default=5000
    )
    parser.add_argument(
        "--max-length", type=int, default=256
    )

    args = parser.parse_args()
    if min(args.batch_size, args.chunk_size, args.max_length) <= 0:
        parser.error("Ukuran batch, chunk, dan token harus positif.")

    OUT = Path(args.output).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.mode == "run":
        predict(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
