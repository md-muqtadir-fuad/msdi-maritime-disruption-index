"""Model-independent semantic event clustering for the pre-LLM news corpus.

The module intentionally separates article deduplication from event matching.
Exact text/URL duplicates should already have been handled upstream.  Here,
independently written articles about the same real-world event retain their
article IDs but receive one shared ``event_id``.  Downstream code can therefore
control repeated media coverage without discarding corroborating sources.

The default encoder is BAAI/bge-m3: a multilingual, long-context embedding
model that is materially stronger than the small E5 encoder used by the legacy
prototype.  The threshold is still a measurement parameter and must be
calibrated on the exported human pair-validation sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import math
import re
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_DAY_WINDOW = 3
DEFAULT_THRESHOLD_CANDIDATES = (0.85, 0.88, 0.90, 0.92)

DATE_CANDIDATES = (
    "date",
    "retrieved_publish_date",
    "publication_date",
    "published_date",
    "publish_date",
)
TITLE_CANDIDATES = ("retrieved_title", "headline", "title", "headline_clean")
TEXT_CANDIDATES = ("retrieved_text", "cleaned_text", "full_text", "text", "content")
SOURCE_CANDIDATES = ("source_domain", "source_name", "source", "publisher")


@dataclass(frozen=True)
class EventClusteringConfig:
    model_name: str = DEFAULT_MODEL
    model_revision: str | None = None
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    window_value: int = DEFAULT_DAY_WINDOW
    window_unit: str = "days"
    # Backward-compatible test/programmatic override. New configurations should
    # use window_value + window_unit so calendar-month windows remain exact.
    day_window: int | None = None
    threshold_candidates: tuple[float, ...] = DEFAULT_THRESHOLD_CANDIDATES
    batch_size: int = 16
    device: str = "auto"
    torch_dtype: str = "auto"
    local_files_only: bool = False
    max_sequence_length: int = 2048
    maximum_body_characters: int = 12000
    pair_audit_min_similarity: float = 0.80
    validation_sample_size: int = 400
    random_seed: int = 20260810

    def validate(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if self.window_value < 0:
            raise ValueError("window_value must be non-negative")
        if self.day_window is not None and self.day_window < 0:
            raise ValueError("day_window must be non-negative")
        if self.window_unit not in {"days", "weeks", "months"}:
            raise ValueError("window_unit must be days, weeks, or months")
        if not self.threshold_candidates:
            raise ValueError("threshold_candidates cannot be empty")
        if any(not 0.0 < value <= 1.0 for value in self.threshold_candidates):
            raise ValueError("every threshold candidate must be in (0, 1]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.torch_dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(
                "torch_dtype must be auto, float16, bfloat16, or float32"
            )
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.maximum_body_characters <= 0:
            raise ValueError("maximum_body_characters must be positive")
        if not 0.0 <= self.pair_audit_min_similarity <= 1.0:
            raise ValueError("pair_audit_min_similarity must be in [0, 1]")
        if self.validation_sample_size < 0:
            raise ValueError("validation_sample_size must be non-negative")

    @property
    def effective_window(self) -> tuple[int, str]:
        if self.day_window is not None:
            return self.day_window, "days"
        return self.window_value, self.window_unit


def clean_string(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = re.sub(r"\s+", " ", str(value)).strip()
    return "" if text.casefold() in {"nan", "none", "null", "n/a"} else text


def threshold_flag_column(value: float) -> str:
    return f"passes_threshold_{value:.2f}".replace(".", "_")


def first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def resolve_event_columns(frame: pd.DataFrame) -> dict[str, str | None]:
    resolved = {
        "article_id": "article_id" if "article_id" in frame.columns else None,
        "date": first_existing(frame, DATE_CANDIDATES),
        "title": first_existing(frame, TITLE_CANDIDATES),
        "text": first_existing(frame, TEXT_CANDIDATES),
        "source": first_existing(frame, SOURCE_CANDIDATES),
    }
    if resolved["article_id"] is None:
        raise ValueError("Event clustering requires an article_id column")
    if resolved["date"] is None:
        raise ValueError(f"No publication-date column found; tried {DATE_CANDIDATES}")
    if resolved["title"] is None and resolved["text"] is None:
        raise ValueError("No usable title or article-text column found")
    return resolved


def parse_event_day(frame: pd.DataFrame, resolved: dict[str, str | None]) -> pd.Series:
    """Use the first parseable candidate date, in declared precedence order."""
    combined = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    ordered = [str(resolved["date"])]
    ordered.extend(column for column in DATE_CANDIDATES if column not in ordered)
    for column in ordered:
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", format="mixed", utc=True)
        parsed = parsed.dt.tz_localize(None).dt.normalize()
        combined = combined.fillna(parsed)
    return combined


def _bounded_body(text: str, maximum_characters: int) -> str:
    if len(text) <= maximum_characters:
        return text
    # Keep the lead, where news reports normally identify the event, and a
    # smaller tail for consequences/context.  The separator makes truncation
    # explicit to the encoder.
    tail = max(1000, maximum_characters // 5)
    head = maximum_characters - tail
    return f"{text[:head]} [TRUNCATED] {text[-tail:]}"


def build_embedding_texts(
    frame: pd.DataFrame,
    resolved: dict[str, str | None],
    maximum_body_characters: int,
) -> list[str]:
    title_column = resolved["title"]
    text_column = resolved["text"]
    texts: list[str] = []
    for _, row in frame.iterrows():
        title = clean_string(row[title_column]) if title_column else ""
        body = clean_string(row[text_column]) if text_column else ""
        body = _bounded_body(body, maximum_body_characters)
        # Do not prepend a shared retrieval instruction: in a symmetric
        # clustering task that repeated text can inflate every pair's cosine.
        value = f"Title: {title}\nArticle: {body}"
        if not title and not body:
            raise ValueError("An article has no usable event-embedding text")
        texts.append(value)
    return texts


def load_embedding_model(config: EventClusteringConfig):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for event clustering; install requirements.txt"
        ) from exc

    kwargs: dict[str, Any] = {"local_files_only": config.local_files_only}
    if config.device != "auto":
        kwargs["device"] = config.device
    if config.torch_dtype != "auto":
        import torch

        kwargs["model_kwargs"] = {
            "torch_dtype": {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[config.torch_dtype]
        }
    if config.model_revision:
        kwargs["revision"] = config.model_revision
    model = SentenceTransformer(config.model_name, **kwargs)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = config.max_sequence_length
    return model


def encode_articles(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    output = np.asarray(embeddings, dtype=np.float32)
    if output.ndim != 2 or output.shape[0] != len(texts):
        raise RuntimeError(
            f"Unexpected embedding shape {output.shape}; expected ({len(texts)}, dimension)"
        )
    norms = np.linalg.norm(output, axis=1)
    if not np.all(np.isfinite(output)) or not np.allclose(norms, 1.0, atol=1e-3):
        raise RuntimeError("Embedding output is non-finite or not L2-normalized")
    return output


def embed_event_articles(
    frame: pd.DataFrame,
    config: EventClusteringConfig,
    *,
    model: Any | None = None,
) -> tuple[np.ndarray, Any]:
    """Embed dated articles once in the canonical order used by every profile."""
    config.validate()
    resolved = resolve_event_columns(frame)
    output = frame.copy().reset_index(drop=True)
    output["article_id"] = output["article_id"].map(clean_string)
    if output["article_id"].eq("").any() or output["article_id"].duplicated().any():
        raise ValueError("Event input must contain one non-empty row per article_id")
    output["event_day"] = parse_event_day(output, resolved)
    valid = output[output["event_day"].notna()].copy()
    valid = valid.sort_values(["event_day", "article_id"], kind="stable").reset_index()
    texts = build_embedding_texts(valid, resolved, config.maximum_body_characters)
    model = model or load_embedding_model(config)
    return encode_articles(model, texts, config.batch_size), model


def _event_identifier(members: list[int], work: pd.DataFrame) -> str:
    earliest = pd.Timestamp(work.loc[members, "event_day"].min()).strftime("%Y%m%d")
    article_ids = sorted(work.loc[members, "article_id"].astype(str).tolist())
    digest = sha256("\n".join(article_ids).encode("utf-8")).hexdigest()[:12]
    return f"EVT_{earliest}_{digest}"


def within_event_window(
    earlier: pd.Timestamp,
    later: pd.Timestamp,
    window_value: int,
    window_unit: str,
) -> bool:
    """Return whether two ordered dates fall inside the configured window."""
    if later < earlier:
        earlier, later = later, earlier
    if window_unit == "days":
        boundary = earlier + pd.Timedelta(days=window_value)
    elif window_unit == "weeks":
        boundary = earlier + pd.Timedelta(weeks=window_value)
    elif window_unit == "months":
        boundary = earlier + pd.DateOffset(months=window_value)
    else:
        raise ValueError(f"Unsupported event window unit: {window_unit}")
    return bool(later <= boundary)


def greedy_complete_linkage(
    work: pd.DataFrame,
    embeddings: np.ndarray,
    threshold: float,
    window_value: int,
    window_unit: str = "days",
) -> list[list[int]]:
    """Deterministic, date-bounded complete-linkage event clustering.

    Rows are processed by publication date and article ID.  An article may join
    a cluster only when it is within the date window and meets the similarity
    threshold against *every* member.  This prevents the transitive chaining
    failure common to connected-component clustering while avoiding a global
    O(n^2) distance matrix.
    """
    clusters: list[list[int]] = []
    cluster_earliest: list[pd.Timestamp] = []
    for row_index in range(len(work)):
        day = pd.Timestamp(work.at[row_index, "event_day"])
        candidates: list[tuple[float, float, int]] = []
        for cluster_index, members in enumerate(clusters):
            if not within_event_window(
                cluster_earliest[cluster_index], day, window_value, window_unit
            ):
                continue
            similarities = embeddings[members] @ embeddings[row_index]
            minimum = float(np.min(similarities))
            if minimum >= threshold:
                candidates.append((minimum, float(np.mean(similarities)), cluster_index))
        if not candidates:
            clusters.append([row_index])
            cluster_earliest.append(day)
            continue
        # Highest weakest-link similarity wins; mean similarity and the earlier
        # deterministic cluster index break ties.
        _, _, selected = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
        clusters[selected].append(row_index)
    return clusters


def greedy_complete_linkage_from_pair_audit(
    work: pd.DataFrame,
    pair_audit: pd.DataFrame,
    threshold: float,
    window_value: int,
    window_unit: str = "days",
    audit_similarity_floor: float = 0.0,
) -> list[list[int]]:
    """Replay deterministic complete-linkage clustering from saved similarities.

    ``build_pair_audit`` retains only pairs whose similarity is at least its
    configured audit floor.  A target threshold at or above that floor can be
    replayed without embeddings: an absent in-window pair is known to be below
    the audit floor and therefore cannot satisfy the target threshold.

    ``work`` must use the same canonical ordering as :func:`greedy_complete_linkage`:
    publication date followed by article ID.  The caller is responsible for
    selecting the appropriate temporal profile from a combined pair audit.
    """
    required_work = {"article_id", "event_day"}
    required_pairs = {"article_id_a", "article_id_b", "cosine_similarity"}
    missing_work = sorted(required_work.difference(work.columns))
    missing_pairs = sorted(required_pairs.difference(pair_audit.columns))
    if missing_work:
        raise ValueError(f"Event work frame lacks required columns: {missing_work}")
    if missing_pairs:
        raise ValueError(f"Pair audit lacks required columns: {missing_pairs}")
    if threshold < audit_similarity_floor:
        raise ValueError(
            "Cannot replay below the saved pair-audit similarity floor: "
            f"threshold={threshold}, audit_floor={audit_similarity_floor}"
        )

    article_ids = work["article_id"].astype(str)
    if article_ids.eq("").any() or article_ids.duplicated().any():
        raise ValueError("Event work frame must contain unique non-empty article IDs")
    days = pd.to_datetime(work["event_day"], errors="coerce")
    if days.isna().any():
        raise ValueError("Pair-audit replay accepts dated articles only")
    expected_order = (
        work.assign(_event_day_order=days, _article_id_order=article_ids)
        .sort_values(["_event_day_order", "_article_id_order"], kind="stable")
        .index.tolist()
    )
    if expected_order != work.index.tolist():
        raise ValueError("Event work frame is not in canonical date/article-ID order")

    pairs = pair_audit[list(required_pairs)].copy()
    pairs["article_id_a"] = pairs["article_id_a"].astype(str)
    pairs["article_id_b"] = pairs["article_id_b"].astype(str)
    pairs["cosine_similarity"] = pd.to_numeric(
        pairs["cosine_similarity"], errors="raise"
    )
    if pairs["cosine_similarity"].lt(audit_similarity_floor).any():
        raise ValueError("Pair audit contains similarities below its declared floor")

    similarity_lookup: dict[tuple[str, str], float] = {}
    for row in pairs.itertuples(index=False):
        left, right = sorted((str(row.article_id_a), str(row.article_id_b)))
        key = (left, right)
        similarity = float(row.cosine_similarity)
        if key in similarity_lookup and not np.isclose(
            similarity_lookup[key], similarity, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"Conflicting saved similarities for article pair {key}")
        similarity_lookup[key] = similarity

    work_ids = set(article_ids)
    pair_ids = set(pairs["article_id_a"]).union(pairs["article_id_b"])
    unknown_ids = sorted(pair_ids.difference(work_ids))
    if unknown_ids:
        raise ValueError(f"Pair audit references unknown article IDs: {unknown_ids[:10]}")

    clusters: list[list[int]] = []
    cluster_earliest: list[pd.Timestamp] = []
    id_by_index = article_ids.to_dict()
    active_cluster_start = 0
    for row_index in range(len(work)):
        day = pd.Timestamp(work.at[row_index, "event_day"])
        row_id = id_by_index[row_index]
        candidates: list[tuple[float, float, int]] = []
        # Cluster creation follows canonical date order, so earliest dates are
        # monotone.  Once a cluster falls outside the window it can never become
        # eligible again for a later article.
        while active_cluster_start < len(clusters) and not within_event_window(
            cluster_earliest[active_cluster_start], day, window_value, window_unit
        ):
            active_cluster_start += 1
        for cluster_index in range(active_cluster_start, len(clusters)):
            members = clusters[cluster_index]
            similarities: list[float] = []
            for member in members:
                member_id = id_by_index[member]
                key = tuple(sorted((member_id, row_id)))
                similarity = similarity_lookup.get(key)
                if similarity is None or similarity < threshold:
                    similarities = []
                    break
                similarities.append(similarity)
            if similarities:
                candidates.append(
                    (min(similarities), float(np.mean(similarities)), cluster_index)
                )
        if not candidates:
            clusters.append([row_index])
            cluster_earliest.append(day)
            continue
        _, _, selected = max(candidates, key=lambda item: (item[0], item[1], -item[2]))
        clusters[selected].append(row_index)
    return clusters


def build_pair_audit(
    work: pd.DataFrame,
    embeddings: np.ndarray,
    window_value: int,
    window_unit: str = "days",
    minimum_similarity: float = 0.0,
) -> pd.DataFrame:
    columns = [
        "article_id_a",
        "article_id_b",
        "event_day_a",
        "event_day_b",
        "day_difference",
        "source_a",
        "source_b",
        "title_a",
        "title_b",
        "text_excerpt_a",
        "text_excerpt_b",
        "cosine_similarity",
        "event_id_a",
        "event_id_b",
        "same_event_cluster",
    ]
    rows: list[dict[str, Any]] = []
    days = work["event_day"].to_numpy(dtype="datetime64[D]")
    for left in range(len(work)):
        for right in range(left + 1, len(work)):
            difference = int((days[right] - days[left]).astype(int))
            if not within_event_window(
                pd.Timestamp(work.at[left, "event_day"]),
                pd.Timestamp(work.at[right, "event_day"]),
                window_value,
                window_unit,
            ):
                break
            similarity = float(embeddings[left] @ embeddings[right])
            if similarity < minimum_similarity:
                continue
            rows.append(
                {
                    "article_id_a": work.at[left, "article_id"],
                    "article_id_b": work.at[right, "article_id"],
                    "event_day_a": pd.Timestamp(work.at[left, "event_day"]).date().isoformat(),
                    "event_day_b": pd.Timestamp(work.at[right, "event_day"]).date().isoformat(),
                    "day_difference": difference,
                    "source_a": work.at[left, "_event_source"],
                    "source_b": work.at[right, "_event_source"],
                    "title_a": work.at[left, "_event_title"],
                    "title_b": work.at[right, "_event_title"],
                    "text_excerpt_a": work.at[left, "_event_excerpt"],
                    "text_excerpt_b": work.at[right, "_event_excerpt"],
                    "cosine_similarity": similarity,
                    "event_id_a": work.at[left, "event_id"],
                    "event_id_b": work.at[right, "event_id"],
                    "same_event_cluster": int(
                        work.at[left, "event_id"] == work.at[right, "event_id"]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def make_pair_validation_sample(
    pair_audit: pd.DataFrame,
    threshold: float,
    sample_size: int,
    random_seed: int,
    threshold_candidates: Sequence[float] | None = None,
) -> pd.DataFrame:
    columns = list(pair_audit.columns) + [
        "sampling_stratum",
        "reviewer_id",
        "human_same_event",
        "reviewer_confidence",
        "review_notes",
    ]
    if pair_audit.empty or sample_size == 0:
        return pd.DataFrame(columns=columns)

    sample_size = min(sample_size, len(pair_audit))
    work = pair_audit.copy()
    thresholds = tuple(sorted(set(threshold_candidates or (threshold,))))
    for candidate in thresholds:
        work[threshold_flag_column(candidate)] = work["cosine_similarity"].ge(candidate).astype(int)

    distances = pd.concat(
        {
            candidate: (work["cosine_similarity"] - candidate).abs()
            for candidate in thresholds
        },
        axis=1,
    )
    work["closest_candidate_threshold"] = distances.idxmin(axis=1).astype(float)
    work["_distance_to_threshold"] = distances.min(axis=1)
    work["sampling_stratum"] = ""

    near_target = int(math.ceil(sample_size * 0.80))
    per_side = max(1, int(math.ceil(near_target / (2 * len(thresholds)))))
    selected_parts = []
    for candidate in thresholds:
        for positive in (True, False):
            candidates = work[work["cosine_similarity"].ge(candidate).eq(positive)].copy()
            candidates["_candidate_distance"] = (
                candidates["cosine_similarity"] - candidate
            ).abs()
            candidates["sampling_stratum"] = (
                f"near_{candidate:.2f}_{'above' if positive else 'below'}"
            )
            selected_parts.append(
                candidates.nsmallest(min(per_side, len(candidates)), "_candidate_distance")
            )
    selected = pd.concat(selected_parts, ignore_index=False).drop_duplicates(
        ["article_id_a", "article_id_b"]
    )
    remaining_n = sample_size - len(selected)
    if remaining_n > 0:
        pool = work.drop(index=selected.index, errors="ignore")
        if not pool.empty:
            random_part = pool.sample(
                n=min(remaining_n, len(pool)), random_state=random_seed
            ).copy()
            random_part["sampling_stratum"] = "random_candidate_pair"
            selected = pd.concat([selected, random_part], ignore_index=False)

    selected = selected.head(sample_size).copy()
    selected["reviewer_id"] = ""
    selected["human_same_event"] = ""
    selected["reviewer_confidence"] = ""
    selected["review_notes"] = ""
    return selected.drop(
        columns=["_distance_to_threshold", "_candidate_distance"], errors="ignore"
    ).reset_index(drop=True)


def cluster_event_articles(
    frame: pd.DataFrame,
    config: EventClusteringConfig,
    *,
    model: Any | None = None,
    embeddings: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return enriched articles, mapping, catalog, pair audit, validation sample, audit."""
    config.validate()
    resolved = resolve_event_columns(frame)
    output = frame.copy().reset_index(drop=True)
    output["article_id"] = output["article_id"].map(clean_string)
    if output["article_id"].eq("").any() or output["article_id"].duplicated().any():
        raise ValueError("Event input must contain one non-empty row per article_id")
    output["event_day"] = parse_event_day(output, resolved)

    valid = output[output["event_day"].notna()].copy()
    valid = valid.sort_values(["event_day", "article_id"], kind="stable").reset_index()
    valid["_event_title"] = (
        valid[str(resolved["title"])].map(clean_string) if resolved["title"] else ""
    )
    valid["_event_source"] = (
        valid[str(resolved["source"])].map(clean_string) if resolved["source"] else ""
    )
    valid["_event_excerpt"] = (
        valid[str(resolved["text"])].map(clean_string).str.slice(0, 1500)
        if resolved["text"]
        else ""
    )
    texts = build_embedding_texts(valid, resolved, config.maximum_body_characters)
    if embeddings is None:
        model = model or load_embedding_model(config)
        embeddings = encode_articles(model, texts, config.batch_size)
    else:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape[0] != len(valid):
            raise ValueError("Synthetic/provided embeddings do not match valid article rows")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1.0, norms)

    window_value, window_unit = config.effective_window
    clusters = greedy_complete_linkage(
        valid,
        embeddings,
        config.similarity_threshold,
        window_value,
        window_unit,
    )
    valid["event_id"] = ""
    for members in clusters:
        event_id = _event_identifier(members, valid)
        valid.loc[members, "event_id"] = event_id

    pair_audit = build_pair_audit(
        valid,
        embeddings,
        window_value,
        window_unit,
        config.pair_audit_min_similarity,
    )
    pair_audit["passes_similarity_threshold"] = pair_audit["cosine_similarity"].ge(
        config.similarity_threshold
    ).astype(int)
    for candidate in sorted(set(config.threshold_candidates)):
        pair_audit[threshold_flag_column(candidate)] = pair_audit[
            "cosine_similarity"
        ].ge(candidate).astype(int)

    event_assignment = valid.set_index("index")["event_id"]
    output["event_id"] = output.index.to_series().map(event_assignment)
    missing_date = output["event_day"].isna()
    for index in output.index[missing_date]:
        digest = sha256(output.at[index, "article_id"].encode("utf-8")).hexdigest()[:12]
        output.at[index, "event_id"] = f"EVT_UNDATED_{digest}"

    output["event_cluster_size"] = output.groupby("event_id")["article_id"].transform("nunique")
    output["event_source_count"] = output.groupby("event_id")[
        str(resolved["source"])
    ].transform("nunique") if resolved["source"] else 1
    event_dates = pd.to_datetime(output["event_day"], errors="coerce")
    output["event_span_days"] = event_dates.groupby(output["event_id"]).transform(
        lambda values: int((values.max() - values.min()).days) if values.notna().any() else 0
    ).astype(int)
    output["event_weight_fixed_total_descriptive"] = 1.0 / output["event_cluster_size"].clip(lower=1)
    output["event_weight_sqrt_descriptive"] = 1.0 / np.sqrt(
        output["event_cluster_size"].clip(lower=1)
    )

    mapping_columns = [
        "article_id",
        "event_day",
        "event_id",
        "event_cluster_size",
        "event_source_count",
        "event_span_days",
        "event_weight_fixed_total_descriptive",
        "event_weight_sqrt_descriptive",
    ]
    event_map = output[mapping_columns].copy()
    event_map["event_day"] = pd.to_datetime(event_map["event_day"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )

    catalog_rows: list[dict[str, Any]] = []
    title_column = str(resolved["title"]) if resolved["title"] else None
    source_column = str(resolved["source"]) if resolved["source"] else None
    for event_id, group in output.groupby("event_id", sort=True):
        dates = pd.to_datetime(group["event_day"], errors="coerce").dropna()
        titles = group[title_column].map(clean_string) if title_column else pd.Series(dtype=str)
        titles = titles[titles.ne("")]
        catalog_rows.append(
            {
                "event_id": event_id,
                "start_day": dates.min().date().isoformat() if not dates.empty else "",
                "end_day": dates.max().date().isoformat() if not dates.empty else "",
                "event_span_days": int(group["event_span_days"].max()),
                "article_count": int(group["article_id"].nunique()),
                "source_count": int(group[source_column].nunique()) if source_column else 0,
                "representative_title": titles.iloc[0] if not titles.empty else "",
                "article_ids": " | ".join(group["article_id"].astype(str)),
            }
        )
    event_catalog = pd.DataFrame(catalog_rows)
    validation_sample = make_pair_validation_sample(
        pair_audit,
        config.similarity_threshold,
        config.validation_sample_size,
        config.random_seed,
        config.threshold_candidates,
    )

    model_commit = None
    if model is not None:
        try:
            model_commit = model[0].auto_model.config._commit_hash
        except (AttributeError, IndexError, TypeError):
            model_commit = None
    audit = {
        "configuration": asdict(config),
        "resolved_columns": resolved,
        "embedding_model_resolved_commit": model_commit,
        "algorithm": "deterministic_date_ordered_greedy_complete_linkage",
        "anti_chaining_rule": "candidate must meet threshold against every cluster member",
        "date_window_interpretation": (
            f"absolute publication difference <= {window_value} {window_unit}"
        ),
        "human_validation_threshold_candidates": list(config.threshold_candidates),
        "counts": {
            "articles": int(len(output)),
            "dated_articles": int(output["event_day"].notna().sum()),
            "undated_singletons": int(output["event_day"].isna().sum()),
            "events": int(output["event_id"].nunique()),
            "multi_article_events": int(event_catalog["article_count"].gt(1).sum()),
            "articles_in_multi_article_events": int(output["event_cluster_size"].gt(1).sum()),
            "candidate_pairs": int(len(pair_audit)),
            "pair_validation_sample": int(len(validation_sample)),
        },
        "guardrails": {
            "llm_scores_used": False,
            "scfi_used": False,
            "articles_removed_by_event_clustering": 0,
            "threshold_requires_human_calibration": True,
            "pair_audit_is_threshold_focused": True,
            "descriptive_global_weights_not_for_realtime_forecasting": True,
        },
    }
    return output, event_map, event_catalog, pair_audit, validation_sample, audit
