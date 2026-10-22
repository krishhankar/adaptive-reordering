"""BM25 and e5 retrieval with reusable JSONL retrieval caches."""

import argparse
import hashlib
import json
import logging
import mmap
import math
import random
import re
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Passage:
    """A retrieved passage and its retrieval score."""

    id: str
    text: str
    score: float


class Retriever:
    """Unified query interface for retrieval backends."""

    def search(self, query: str, k: int) -> list[Passage]:
        raise NotImplementedError


class BM25Retriever(Retriever):
    """Pyserini BM25 wrapper."""

    def __init__(
        self,
        index: str,
        corpus_subset: str | None = None,
        subset_size: int | None = None,
        seed: int = 42,
        index_root: str | Path = "indexes",
    ) -> None:
        if corpus_subset:
            subset_path = build_bm25_subset(
                corpus_subset, subset_size, seed, index_root
            )
            self.local_index = LocalBM25Retriever(subset_path)
            self.searcher = None
            return
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as error:
            raise RuntimeError("Install pyserini to use BM25 retrieval.") from error
        self.searcher = (
            LuceneSearcher.from_prebuilt_index(index)
            if not Path(index).exists()
            else LuceneSearcher(index)
        )
        self.local_index = None

    def search(self, query: str, k: int) -> list[Passage]:
        if self.local_index is not None:
            return self.local_index.search(query, k)
        hits = self.searcher.search(query, k)
        output = []
        for hit in hits:
            document = self.searcher.doc(hit.docid)
            if document is None:
                raise RuntimeError(f"BM25 index did not return document {hit.docid}.")
            raw = document.raw()
            text = ""
            if raw:
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError:
                    text = raw
                else:
                    if isinstance(decoded, dict):
                        text = decoded.get("contents") or decoded.get("text") or ""
            if not text:
                text = document.contents() or ""
            if not text:
                raise RuntimeError(
                    f"BM25 document {hit.docid} has no stored passage text."
                )
            output.append(Passage(hit.docid, text, float(hit.score)))
        return output


class LocalBM25Retriever(Retriever):
    """Small in-memory BM25 index for offline development subsets."""

    def __init__(self, corpus_path: str | Path, k1: float = 1.2, b: float = 0.75):
        self.documents: list[Passage] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.k1 = k1
        self.b = b
        with Path(corpus_path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = str(record.get("contents", record.get("text", "")))
                terms = Counter(re.findall(r"\w+", text.lower()))
                self.documents.append(
                    Passage(
                        str(record.get("id", line_number)),
                        text,
                        0.0,
                    )
                )
                self.term_frequencies.append(terms)
                self.document_frequencies.update(terms.keys())
        if not self.documents:
            raise ValueError(f"Local BM25 subset is empty: {corpus_path}")
        self.lengths = [sum(terms.values()) for terms in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)

    def search(self, query: str, k: int) -> list[Passage]:
        query_terms = set(re.findall(r"\w+", query.lower()))
        total_documents = len(self.documents)
        scored: list[tuple[float, int]] = []
        for index, terms in enumerate(self.term_frequencies):
            score = 0.0
            document_length = self.lengths[index]
            for term in query_terms:
                frequency = terms.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self.document_frequencies[term]
                inverse_document_frequency = math.log(
                    1
                    + (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * document_length / self.average_length
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                )
            scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            Passage(
                self.documents[index].id,
                self.documents[index].text,
                score,
            )
            for score, index in scored[:k]
        ]


class E5Retriever(Retriever):
    """SentenceTransformers e5 encoder backed by a FAISS index."""

    def __init__(self, index_dir: str, model_name: str, device: str = "cuda") -> None:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Install faiss-cpu/faiss-gpu and sentence-transformers for e5."
            ) from error
        self.faiss = faiss
        self.index = faiss.read_index(str(Path(index_dir) / "index.faiss"))
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = 16
        self.encoder = SentenceTransformer(model_name, device=device)
        self.metadata_path = Path(index_dir) / "passages.jsonl"
        self.offsets_path = Path(index_dir) / "passages.offsets"
        self._metadata_stream = self.metadata_path.open("rb")
        self._offsets_stream = self.offsets_path.open("rb")
        self._offsets = mmap.mmap(
            self._offsets_stream.fileno(), 0, access=mmap.ACCESS_READ
        )
        if self.index.ntotal != len(self._offsets) // 8:
            raise ValueError("FAISS index and passage metadata contain different counts.")

    def search(self, query: str, k: int) -> list[Passage]:
        vector = self.encoder.encode(
            [f"query: {query}"], normalize_embeddings=True
        )
        scores, indices = self.index.search(vector, k)
        output = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            offset = struct.unpack_from("<Q", self._offsets, int(index) * 8)[0]
            self._metadata_stream.seek(offset)
            record = json.loads(self._metadata_stream.readline())
            output.append(
                Passage(
                    id=str(record["id"]),
                    text=str(record["text"]),
                    score=float(score),
                )
            )
        return output


def build_bm25_subset(
    corpus_path: str | None,
    subset_size: int | None = 200_000,
    seed: int = 42,
    index_root: str | Path = "indexes",
) -> str:
    """Persist a seeded local corpus sample for the lightweight BM25 dev index."""
    if corpus_path is None:
        raise ValueError("--corpus_subset requires a local --corpus_file.")
    source = Path(corpus_path)
    if not source.is_file():
        raise FileNotFoundError(f"Local corpus file not found: {source}")
    if subset_size is not None and subset_size <= 0:
        raise ValueError("--corpus_subset_size must be positive.")
    effective_size = subset_size or 200_000
    source_stat = source.stat()
    source_fingerprint = hashlib.sha256(
        f"{source.resolve()}:{source_stat.st_size}:{source_stat.st_mtime_ns}".encode()
    ).hexdigest()[:8]
    index_dir = Path(index_root) / (
        f"bm25-subset-{source.stem}-{source_fingerprint}-{effective_size}-seed{seed}"
    )
    jsonl = index_dir / "corpus.jsonl"
    if jsonl.exists():
        return str(jsonl)
    index_dir.mkdir(parents=True, exist_ok=True)
    sample_size = effective_size
    rng = random.Random(seed)
    sample: list[dict[str, str]] = []
    row_count = 0
    with source.open(encoding="utf-8") as input_stream:
        for line_number, line in enumerate(input_stream):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) >= 3 and line_number == 0 and fields[0].lower() == "id":
                continue
            if len(fields) >= 3:
                identifier, title, text = fields
                contents = f"{title} {text}".strip()
            else:
                record = json.loads(line)
                identifier = str(record.get("id", line_number))
                contents = str(record.get("text", record.get("contents", "")))
            row = {"id": identifier, "contents": contents}
            if len(sample) < sample_size:
                sample.append(row)
            else:
                replacement = rng.randrange(row_count + 1)
                if replacement < sample_size:
                    sample[replacement] = row
            row_count += 1
    if not sample:
        raise ValueError(f"Local corpus file is empty: {source}")
    with jsonl.open("w", encoding="utf-8") as output_stream:
        for row in sample:
            output_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(jsonl)


def create_retriever(
    name: str,
    config: dict[str, Any],
    paths: dict[str, Any],
    corpus_subset: bool = False,
    corpus_file: str | None = None,
    corpus_subset_size: int | None = None,
    seed: int = 42,
) -> Retriever:
    """Construct a configured retriever."""
    if name == "bm25":
        return BM25Retriever(
            str(paths["bm25_index"]),
            corpus_subset=corpus_file if corpus_subset else None,
            subset_size=corpus_subset_size if corpus_subset else None,
            seed=seed,
            index_root=str(paths["indexes"]),
        )
    if name == "e5":
        return E5Retriever(
            str(paths["e5_index"]),
            str(config["model"]),
            str(config.get("device", "cuda")),
        )
    raise ValueError(f"Unsupported retriever: {name}")


def write_retrieval_cache(
    examples: list[dict[str, Any]],
    retriever: Retriever,
    output_path: str | Path,
    k: int = 200,
) -> None:
    """Retrieve once per question and persist its top-k passages to JSONL."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for example in examples:
            passages = retriever.search(example["question"], k)
            stream.write(
                json.dumps(
                    {
                        "id": example["id"],
                        "passages": [asdict(passage) for passage in passages],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("Wrote retrieval cache: %s", path)


def save_retrieval_cache(
    records: dict[str, list[Passage]], output_path: str | Path
) -> None:
    """Persist already retrieved passages without issuing new queries."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for identifier, passages in records.items():
            stream.write(
                json.dumps(
                    {
                        "id": identifier,
                        "passages": [asdict(passage) for passage in passages],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_retrieval_cache(path: str | Path) -> dict[str, list[Passage]]:
    """Read a retrieval cache, rejecting malformed or duplicate question IDs."""
    records: dict[str, list[Passage]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = str(record["id"])
            if identifier in records:
                raise ValueError(f"Duplicate retrieval id {identifier!r} at line {line_number}.")
            records[identifier] = [
                Passage(str(item["id"]), str(item["text"]), float(item["score"]))
                for item in record["passages"]
            ]
    return records


def main() -> None:
    """Build retrieval cache using config-selected backends."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("nq", "popqa"))
    parser.add_argument("--retriever", required=True, choices=("bm25", "e5"))
    parser.add_argument("--dataset_file")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--corpus_subset", action="store_true")
    parser.add_argument("--corpus_file")
    parser.add_argument("--corpus_subset_size", type=int, default=200_000)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import yaml

    from src.data import load_dataset

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_config = config["datasets"][args.dataset]
    examples = load_dataset(
        args.dataset,
        dataset_config["hf_name"],
        dataset_config["split"],
        args.limit,
        args.dataset_file,
    )
    cache_path = Path(config["paths"]["retrieval_cache"]) / f"{args.dataset}_{args.retriever}.jsonl"
    records = (
        load_retrieval_cache(cache_path)
        if cache_path.exists() and not args.refresh
        else {}
    )
    missing = [example for example in examples if example["id"] not in records]
    if missing:
        retriever = create_retriever(
            args.retriever,
            config["retrievers"][args.retriever],
            config["paths"],
            args.corpus_subset,
            args.corpus_file,
            args.corpus_subset_size,
            int(config["seed"]),
        )
        temporary_cache = cache_path.with_suffix(".missing.jsonl")
        write_retrieval_cache(
            missing,
            retriever,
            temporary_cache,
            int(config["retrieval"]["max_passages"]),
        )
        records.update(load_retrieval_cache(temporary_cache))
        temporary_cache.unlink()
        save_retrieval_cache(records, cache_path)
    else:
        logger.info("Retrieval cache already covers all requested questions: %s", cache_path)


if __name__ == "__main__":
    main()
