"""Build a compressed FAISS index for the DPR psgs_w100 passage corpus."""

import argparse
import json
import logging
import struct
from pathlib import Path
from collections.abc import Iterator

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _passages(corpus_path: Path) -> Iterator[tuple[str, str]]:
    with corpus_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t", 2)
            if line_number == 0 and len(fields) >= 3 and fields[0].lower() == "id":
                continue
            if len(fields) < 3:
                raise ValueError(
                    f"Expected DPR TSV columns (id, title, text) at line {line_number + 1}."
                )
            identifier, title, text = fields
            yield identifier, f"{title} {text}".strip()


def build_index(
    corpus_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    device: str,
) -> None:
    """Encode passages in batches and add them to a non-learning FAISS index."""
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = SentenceTransformer(model_name, device=device)
    dimension = encoder.get_sentence_embedding_dimension()
    index = faiss.IndexScalarQuantizer(
        dimension,
        faiss.ScalarQuantizer.QT_fp16,
        faiss.METRIC_INNER_PRODUCT,
    )

    metadata_path = output_dir / "passages.jsonl"
    offsets_path = output_dir / "passages.offsets"
    with metadata_path.open("wb") as metadata_stream, offsets_path.open(
        "wb"
    ) as offsets_stream:
        batch: list[tuple[str, str]] = []

        def index_batch(passages: list[tuple[str, str]]) -> None:
            vectors = encoder.encode(
                [f"passage: {text}" for _, text in passages],
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            index.add(vectors)
            for identifier, text in passages:
                encoded = (
                    json.dumps(
                        {"id": identifier, "text": text}, ensure_ascii=False
                    )
                    + "\n"
                ).encode("utf-8")
                offsets_stream.write(struct.pack("<Q", metadata_stream.tell()))
                metadata_stream.write(encoded)

        passage_iterator = tqdm(_passages(corpus_path), desc="Indexing passages")
        count = 0
        for passage in passage_iterator:
            batch.append(passage)
            if len(batch) == batch_size:
                index_batch(batch)
                count += len(batch)
                batch.clear()
        if batch:
            index_batch(batch)
            count += len(batch)
    faiss.write_index(index, str(output_dir / "index.faiss"))
    logger.info("Indexed %s passages in %s", count, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus/psgs_w100.tsv")
    parser.add_argument("--output", default="indexes/e5")
    parser.add_argument("--model", default="intfloat/e5-large-v2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_index(
        Path(args.corpus),
        Path(args.output),
        args.model,
        args.batch_size,
        args.device,
    )


if __name__ == "__main__":
    main()
