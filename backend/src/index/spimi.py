from dataclasses import dataclass
from typing import Any

from backend.src.index.inverted_index import Posting
from backend.src.index.models import validate_histogram_record


@dataclass(frozen=True)
class SpimiBlock:
    """Partial inverted index created by SPIMI."""

    block_id: int
    postings: dict[int, list[Posting]]

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of the block."""
        return {
            "block_id": self.block_id,
            "postings": {
                codeword: [
                    {"chunk_id": item.chunk_id, "frequency": item.frequency}
                    for item in postings
                ]
                for codeword, postings in self.postings.items()
            },
        }


class SpimiIndexer:
    """Create and merge SPIMI blocks from text codeword histograms."""

    def __init__(self, block_size: int = 1000) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be greater than zero")
        self.block_size = block_size

    def create_blocks(self, records: list[dict[str, Any]]) -> list[SpimiBlock]:
        """Create partial inverted-index blocks in one pass."""
        blocks: list[SpimiBlock] = []
        current: dict[int, list[Posting]] = {}
        records_in_block = 0

        for record in records:
            histogram_record = validate_histogram_record(record)
            if histogram_record.modality != "text":
                raise ValueError("SPIMI only accepts text histograms")

            self._add_histogram(
                current,
                histogram_record.chunk_id,
                histogram_record.histogram,
            )
            records_in_block += 1
            if records_in_block == self.block_size:
                blocks.append(self._build_block(len(blocks), current))
                current = {}
                records_in_block = 0

        if current:
            blocks.append(self._build_block(len(blocks), current))
        return blocks

    def merge_blocks(self, blocks: list[SpimiBlock]) -> dict[int, list[Posting]]:
        """Merge SPIMI blocks into one inverted index posting map."""
        merged: dict[int, list[Posting]] = {}
        for block in sorted(blocks, key=lambda item: item.block_id):
            for codeword, postings in block.postings.items():
                merged.setdefault(codeword, []).extend(postings)

        return {
            codeword: sorted(postings, key=lambda item: item.chunk_id)
            for codeword, postings in sorted(merged.items())
        }

    def _add_histogram(
        self,
        postings: dict[int, list[Posting]],
        chunk_id: str,
        histogram: Any,
    ) -> None:
        for codeword, frequency in enumerate(histogram):
            if frequency > 0:
                postings.setdefault(codeword, []).append(
                    Posting(chunk_id, float(frequency))
                )

    def _build_block(
        self,
        block_id: int,
        postings: dict[int, list[Posting]],
    ) -> SpimiBlock:
        return SpimiBlock(
            block_id=block_id,
            postings={codeword: list(items) for codeword, items in postings.items()},
        )
