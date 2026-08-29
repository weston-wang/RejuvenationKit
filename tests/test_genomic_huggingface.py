from collections.abc import Sequence

import numpy as np
import pytest
from pydantic import ValidationError

import rejuvenationkit.genomics.huggingface as huggingface_module
from rejuvenationkit.genomics import (
    EmbeddingBatch,
    EmbeddingProvenance,
    HFGenomeEmbedder,
    HFGenomeEncoderConfig,
    OverlengthPolicy,
    SequenceWindow,
    StrandPolicy,
)

REVISION = "a" * 40


class CountBackend:
    def encode(self, sequences: Sequence[str], config: HFGenomeEncoderConfig) -> np.ndarray:
        del config
        return np.asarray(
            [
                [
                    sequence.count("A"),
                    sequence.count("C"),
                    sequence.count("G"),
                    sequence.count("T"),
                    len(sequence),
                ]
                for sequence in sequences
            ],
            dtype=float,
        )


class BadBackend:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def encode(self, sequences: Sequence[str], config: HFGenomeEncoderConfig) -> np.ndarray:
        del sequences, config
        return self.values


def config(**updates: object) -> HFGenomeEncoderConfig:
    values: dict[str, object] = {
        "model_id": "test/genome-model",
        "model_revision": REVISION,
        "weights_license": "Apache-2.0",
        "layer": -2,
        "maximum_bases": 16,
        "strand_policy": StrandPolicy.FORWARD,
    }
    values.update(updates)
    return HFGenomeEncoderConfig(**values)


def test_offline_backend_produces_deterministic_provenance_without_raw_sequence() -> None:
    windows = (
        SequenceWindow(sequence_id="promoter-1", sequence="AACCGGTT"),
        SequenceWindow(sequence_id="promoter-2", sequence="AAAATTTT"),
    )
    embedder = HFGenomeEmbedder(config(), backend=CountBackend())
    first = embedder.embed(windows)
    second = embedder.embed(windows)

    assert np.array_equal(first.values, second.values)
    assert first.values.shape == (2, 5)
    assert first.provenance.model_revision == REVISION
    assert first.provenance.input_hashes["promoter-1"] != windows[0].sequence
    assert len(first.provenance.input_hashes["promoter-1"]) == 64
    assert not first.values.flags.writeable
    with pytest.raises(TypeError):
        first.provenance.input_hashes["promoter-1"] = "f" * 64  # type: ignore[index]
    with pytest.raises(TypeError):
        first.provenance.library_versions["transformers"] = "forged"  # type: ignore[index]
    assert (
        EmbeddingProvenance.model_validate(first.provenance.model_dump(mode="python"))
        == first.provenance
    )

    mismatched = first.provenance.model_copy(update={"input_hashes": {"different": "f" * 64}})
    with pytest.raises(ValueError, match="exactly match sequence_ids"):
        EmbeddingBatch(
            sequence_ids=first.sequence_ids,
            values=first.values,
            provenance=mismatched,
        )
    invalid_revision = first.provenance.model_copy(update={"model_revision": "main"})
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        EmbeddingBatch(
            sequence_ids=first.sequence_ids,
            values=first.values,
            provenance=invalid_revision,
        )


def test_forward_reverse_mean_is_orientation_invariant() -> None:
    embedder = HFGenomeEmbedder(
        config(strand_policy=StrandPolicy.FORWARD_REVERSE_MEAN),
        backend=CountBackend(),
    )
    forward = SequenceWindow(sequence_id="forward", sequence="AAAACCGT")
    reverse = SequenceWindow(sequence_id="reverse", sequence="ACGGTTTT")
    result = embedder.embed((forward, reverse))

    assert np.allclose(result.values[0], result.values[1])


def test_chunked_forward_reverse_mean_is_orientation_invariant() -> None:
    embedder = HFGenomeEmbedder(
        config(
            strand_policy=StrandPolicy.FORWARD_REVERSE_MEAN,
            overlength_policy=OverlengthPolicy.CHUNK_MEAN,
            maximum_bases=16,
            chunk_overlap_bases=4,
        ),
        backend=CountBackend(),
    )
    sequence = "CCCCAAAATTTTCCCCGGGG"
    result = embedder.embed(
        (
            SequenceWindow(sequence_id="forward", sequence=sequence),
            SequenceWindow(sequence_id="reverse", sequence="CCCCGGGGAAAATTTTGGGG"),
        )
    )

    assert np.allclose(result.values[0], result.values[1])


def test_overlength_policy_never_silently_truncates() -> None:
    window = SequenceWindow(sequence_id="long", sequence="A" * 20)
    with pytest.raises(ValueError, match="exceeding"):
        HFGenomeEmbedder(config(), backend=CountBackend()).embed((window,))

    chunked = HFGenomeEmbedder(
        config(
            overlength_policy=OverlengthPolicy.CHUNK_MEAN,
            chunk_overlap_bases=4,
        ),
        backend=CountBackend(),
    ).embed((window,))
    assert chunked.values.shape == (1, 5)
    assert chunked.provenance.overlength_policy is OverlengthPolicy.CHUNK_MEAN


def test_overlapping_chunk_pooling_is_fail_visible_as_a_context_heuristic() -> None:
    class AdenineFractionBackend:
        def encode(
            self,
            sequences: Sequence[str],
            config: HFGenomeEncoderConfig,
        ) -> np.ndarray:
            del config
            return np.asarray(
                [[sequence.count("A") / len(sequence)] for sequence in sequences],
                dtype=float,
            )

    sequence = "CCCCCCCCAAAAAAAACCCCCCCC"
    result = HFGenomeEmbedder(
        config(
            overlength_policy=OverlengthPolicy.CHUNK_MEAN,
            maximum_bases=16,
            chunk_overlap_bases=8,
        ),
        backend=AdenineFractionBackend(),
    ).embed((SequenceWindow(sequence_id="overlap-counterexample", sequence=sequence),))

    assert result.values[0, 0] == pytest.approx(0.5)
    assert result.values[0, 0] != pytest.approx(sequence.count("A") / len(sequence))
    assert result.provenance.warnings == ("overlapping_chunk_pooling_is_a_context_heuristic",)
    forged = result.provenance.model_copy(update={"warnings": ()})
    with pytest.raises(ValidationError, match="context-heuristic warning"):
        EmbeddingBatch(
            sequence_ids=result.sequence_ids,
            values=result.values,
            provenance=forged,
        )


def test_variant_embedding_returns_alternate_minus_reference() -> None:
    embedder = HFGenomeEmbedder(config(), backend=CountBackend())
    reference = SequenceWindow(sequence_id="variant-ref", sequence="AAAACCCC")
    alternate = SequenceWindow(sequence_id="variant-alt", sequence="AAAAGCCC")
    result = embedder.embed_variant("1:5:C:G", reference, alternate)

    assert result.variant_id == "1:5:C:G"
    assert np.array_equal(result.delta, result.alternate - result.reference)
    assert result.delta[1] == -1
    assert result.delta[2] == 1


def test_embedding_rejects_duplicate_ids_bad_shapes_and_nonfinite_values() -> None:
    duplicated = (
        SequenceWindow(sequence_id="same", sequence="AAAA"),
        SequenceWindow(sequence_id="same", sequence="CCCC"),
    )
    with pytest.raises(ValueError, match="unique"):
        HFGenomeEmbedder(config(), backend=CountBackend()).embed(duplicated)
    with pytest.raises(ValueError, match="invalid embedding shape"):
        HFGenomeEmbedder(config(), backend=BadBackend(np.ones((1, 2)))).embed(
            (
                SequenceWindow(sequence_id="a", sequence="AAAA"),
                SequenceWindow(sequence_id="b", sequence="CCCC"),
            )
        )
    with pytest.raises(ValueError, match="non-finite"):
        HFGenomeEmbedder(config(), backend=BadBackend(np.asarray([[float("nan")]]))).embed(
            (SequenceWindow(sequence_id="a", sequence="AAAA"),)
        )


def test_sequence_and_model_configuration_are_reproducibility_strict() -> None:
    with pytest.raises(ValidationError, match="only A, C, G, T, and N"):
        SequenceWindow(sequence_id="bad", sequence="ACGTX")
    with pytest.raises(ValidationError, match="match its genomic interval"):
        SequenceWindow(
            sequence_id="bad-coordinates",
            sequence="ACGT",
            genome_assembly="CanFam4",
            chromosome="1",
            start=10,
            end=20,
        )
    with pytest.raises(ValidationError, match="commit SHAs"):
        config(model_revision="main")
    with pytest.raises(ValidationError, match="smaller"):
        config(
            overlength_policy=OverlengthPolicy.CHUNK_MEAN,
            chunk_overlap_bases=16,
        )
    with pytest.raises(ValidationError, match="only valid"):
        config(chunk_overlap_bases=2)


def test_variant_embedding_rejects_unaligned_or_duplicated_windows() -> None:
    embedder = HFGenomeEmbedder(config(), backend=CountBackend())
    with pytest.raises(ValueError, match="equal-length"):
        embedder.embed_variant(
            "variant",
            SequenceWindow(sequence_id="ref", sequence="AAAA"),
            SequenceWindow(sequence_id="alt", sequence="AAA"),
        )
    with pytest.raises(ValueError, match="one SNV"):
        embedder.embed_variant(
            "variant",
            SequenceWindow(sequence_id="ref", sequence="AAAA"),
            SequenceWindow(sequence_id="alt", sequence="CCAA"),
        )
    with pytest.raises(ValueError, match="exactly one SNV"):
        embedder.embed_variant(
            "variant",
            SequenceWindow(sequence_id="ref", sequence="AAAA"),
            SequenceWindow(sequence_id="alt", sequence="AAAA"),
        )
    with pytest.raises(ValueError, match="reference allele"):
        embedder.embed_variant(
            "variant",
            SequenceWindow(
                sequence_id="ref",
                sequence="AAAA",
                reference_allele="C",
                alternate_allele="T",
            ),
            SequenceWindow(sequence_id="alt", sequence="AAAT"),
        )
    with pytest.raises(ValueError, match="distinct"):
        embedder.embed_variant(
            "variant",
            SequenceWindow(sequence_id="same", sequence="AAAA"),
            SequenceWindow(sequence_id="same", sequence="AAAT"),
        )


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HFGenomeEmbedder(config(), backend=CountBackend()).embed(())


def test_default_backend_reports_actionable_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(name: str) -> object:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(huggingface_module, "import_module", missing_import)
    embedder = HFGenomeEmbedder(config())

    with pytest.raises(ImportError, match=r"rejuvenationkit\[genome-hf\]"):
        embedder.embed((SequenceWindow(sequence_id="one", sequence="AAAA"),))
