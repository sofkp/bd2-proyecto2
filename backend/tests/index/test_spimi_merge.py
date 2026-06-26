from backend.src.index.spimi import SpimiIndexer


def test_merge_blocks_combines_postings_by_codeword() -> None:
    """SPIMI merge should group postings from every block."""
    indexer = SpimiIndexer(block_size=1)
    blocks = indexer.create_blocks(
        [
            {"chunk_id": "b", "modality": "text", "histogram": [1, 0]},
            {"chunk_id": "a", "modality": "text", "histogram": [2, 3]},
        ]
    )

    merged = indexer.merge_blocks(blocks)

    assert [posting.chunk_id for posting in merged[0]] == ["a", "b"]
    assert [posting.frequency for posting in merged[0]] == [2.0, 1.0]
    assert merged[1][0].chunk_id == "a"
    assert merged[1][0].frequency == 3.0


def test_merge_blocks_sorts_codewords() -> None:
    """Merged posting maps should expose codewords in sorted order."""
    indexer = SpimiIndexer(block_size=1)
    blocks = indexer.create_blocks(
        [{"chunk_id": "a", "modality": "text", "histogram": [0, 1, 2]}]
    )

    merged = indexer.merge_blocks(blocks)

    assert list(merged) == [1, 2]


def test_merge_blocks_accepts_empty_input() -> None:
    """Merging no blocks should return an empty posting map."""
    assert SpimiIndexer().merge_blocks([]) == {}
