import pytest

from backend.src.index.spimi import SpimiIndexer


def test_create_blocks_splits_records_by_block_size() -> None:
    """SPIMI should create one block per configured record batch."""
    records = [
        {"chunk_id": "a", "modality": "text", "histogram": [1, 0]},
        {"chunk_id": "b", "modality": "text", "histogram": [0, 2]},
        {"chunk_id": "c", "modality": "text", "histogram": [3, 0]},
    ]

    blocks = SpimiIndexer(block_size=2).create_blocks(records)

    assert [block.block_id for block in blocks] == [0, 1]
    assert blocks[0].postings[0][0].chunk_id == "a"
    assert blocks[0].postings[1][0].chunk_id == "b"
    assert blocks[1].postings[0][0].chunk_id == "c"


def test_create_blocks_ignores_zero_frequencies() -> None:
    """Only positive frequencies should become postings."""
    records = [{"chunk_id": "a", "modality": "text", "histogram": [0, 4, 0]}]

    blocks = SpimiIndexer(block_size=10).create_blocks(records)

    assert list(blocks[0].postings) == [1]
    assert blocks[0].postings[1][0].frequency == 4.0


def test_create_blocks_rejects_non_text_records() -> None:
    """SPIMI text indexing should reject non-text histograms."""
    records = [{"chunk_id": "img", "modality": "image", "histogram": [1, 0]}]

    with pytest.raises(ValueError, match="text histograms"):
        SpimiIndexer().create_blocks(records)


def test_create_blocks_returns_empty_list_for_no_records() -> None:
    """No input records should produce no SPIMI blocks."""
    assert SpimiIndexer().create_blocks([]) == []


def test_spimi_block_to_dict_is_serializable() -> None:
    """SPIMI blocks should expose a simple serializable shape."""
    records = [{"chunk_id": "a", "modality": "text", "histogram": [2]}]
    block = SpimiIndexer().create_blocks(records)[0]

    assert block.to_dict() == {
        "block_id": 0,
        "postings": {0: [{"chunk_id": "a", "frequency": 2.0}]},
    }


def test_spimi_indexer_rejects_invalid_block_size() -> None:
    """Block size must be positive."""
    with pytest.raises(ValueError, match="block_size"):
        SpimiIndexer(block_size=0)
