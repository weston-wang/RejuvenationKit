"""Optional, reproducible Hugging Face genome-sequence embeddings."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SEQUENCE_PATTERN = re.compile(r"^[ACGTN]+$")
_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


class PoolingStrategy(StrEnum):
    """Token pooling policy recorded in embedding provenance."""

    MASKED_MEAN = "masked_mean"


class StrandPolicy(StrEnum):
    """Handling of DNA strand orientation."""

    FORWARD = "forward"
    FORWARD_REVERSE_MEAN = "forward_reverse_mean"


class OverlengthPolicy(StrEnum):
    """Handling of sequences longer than the configured base window."""

    ERROR = "error"
    CHUNK_MEAN = "chunk_mean"


class HFModelClass(StrEnum):
    """Transformers auto-model factory selected for a pinned checkpoint."""

    AUTO_MODEL = "auto_model"
    AUTO_MODEL_FOR_MASKED_LM = "auto_model_for_masked_lm"


class SequenceWindow(BaseModel):
    """A DNA sequence window with optional zero-based genomic coordinates."""

    model_config = ConfigDict(frozen=True)

    sequence_id: str = Field(min_length=1)
    sequence: str = Field(min_length=1)
    genome_assembly: str | None = None
    chromosome: str | None = None
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, gt=0)
    reference_allele: str | None = None
    alternate_allele: str | None = None

    @model_validator(mode="after")
    def validate_sequence_and_coordinates(self) -> Self:
        """Normalize accepted symbols and reject ambiguous coordinates."""
        normalized = self.sequence.upper()
        if _SEQUENCE_PATTERN.fullmatch(normalized) is None:
            raise ValueError("sequence may contain only A, C, G, T, and N")
        object.__setattr__(self, "sequence", normalized)
        supplied = sum(value is not None for value in (self.chromosome, self.start, self.end))
        if supplied not in (0, 3):
            raise ValueError("chromosome, start, and end must be supplied together")
        if self.start is not None and self.end is not None:
            if self.end <= self.start:
                raise ValueError("sequence-window end must be greater than start")
            if self.end - self.start != len(normalized):
                raise ValueError("sequence length must match its genomic interval")
            if self.genome_assembly is None:
                raise ValueError("genome_assembly is required with coordinates")
        alleles = (self.reference_allele, self.alternate_allele)
        if sum(value is not None for value in alleles) not in (0, 2):
            raise ValueError("reference and alternate alleles must be supplied together")
        if self.reference_allele is not None and self.alternate_allele is not None:
            reference = self.reference_allele.upper()
            alternate = self.alternate_allele.upper()
            if (
                _SEQUENCE_PATTERN.fullmatch(reference) is None
                or _SEQUENCE_PATTERN.fullmatch(alternate) is None
            ):
                raise ValueError("alleles may contain only A, C, G, T, and N")
            object.__setattr__(self, "reference_allele", reference)
            object.__setattr__(self, "alternate_allele", alternate)
        return self


class HFGenomeEncoderConfig(BaseModel):
    """Immutable Hugging Face model, revision, pooling, and sequence policy."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(min_length=1)
    model_revision: str
    tokenizer_revision: str | None = None
    weights_license: str = Field(min_length=1)
    layer: int
    model_class: HFModelClass = HFModelClass.AUTO_MODEL
    pooling: PoolingStrategy = PoolingStrategy.MASKED_MEAN
    strand_policy: StrandPolicy = StrandPolicy.FORWARD_REVERSE_MEAN
    overlength_policy: OverlengthPolicy = OverlengthPolicy.ERROR
    maximum_bases: int = Field(default=1_000, ge=16)
    chunk_overlap_bases: int = Field(default=0, ge=0)
    maximum_tokens: int = Field(default=2_048, ge=8)
    batch_size: int = Field(default=8, ge=1)
    device: str = Field(default="cpu", min_length=1)
    trust_remote_code: bool = False
    local_files_only: bool = False

    @model_validator(mode="after")
    def validate_reproducibility_and_windows(self) -> Self:
        """Require immutable revisions and a valid overlapping-chunk policy."""
        revisions = (self.model_revision, self.tokenizer_revision or self.model_revision)
        if any(_REVISION_PATTERN.fullmatch(item) is None for item in revisions):
            raise ValueError("model and tokenizer revisions must be full 40-character commit SHAs")
        if self.chunk_overlap_bases >= self.maximum_bases:
            raise ValueError("chunk_overlap_bases must be smaller than maximum_bases")
        if self.overlength_policy is OverlengthPolicy.ERROR and self.chunk_overlap_bases != 0:
            raise ValueError("chunk overlap is only valid with chunk_mean overlength policy")
        return self


class EmbeddingProvenance(BaseModel):
    """Model-card, software, pooling, and input fingerprints for embeddings."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    model_revision: str
    tokenizer_revision: str
    weights_license: str
    layer: int
    model_class: HFModelClass
    pooling: PoolingStrategy
    strand_policy: StrandPolicy
    overlength_policy: OverlengthPolicy
    maximum_bases: int
    chunk_overlap_bases: int
    maximum_tokens: int
    batch_size: int
    device: str
    trust_remote_code: bool
    local_files_only: bool
    library_versions: dict[str, str]
    input_hashes: dict[str, str]


EmbeddingValues = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Sequence embeddings kept separate from calibrated efficacy estimates."""

    sequence_ids: tuple[str, ...]
    values: EmbeddingValues
    provenance: EmbeddingProvenance

    def __post_init__(self) -> None:
        """Validate alignment and freeze finite embedding values."""
        array = np.asarray(self.values, dtype=float).copy()
        if array.ndim != 2 or array.shape[0] != len(self.sequence_ids):
            raise ValueError("embedding matrix must align rows to sequence_ids")
        if len(set(self.sequence_ids)) != len(self.sequence_ids):
            raise ValueError("sequence_ids must be unique")
        if not np.isfinite(array).all():
            raise ValueError("embedding values must be finite")
        array.flags.writeable = False
        object.__setattr__(self, "values", array)


@dataclass(frozen=True, slots=True)
class VariantEmbedding:
    """Reference, alternate, and delta embeddings for one sequence context."""

    variant_id: str
    reference: EmbeddingValues
    alternate: EmbeddingValues
    delta: EmbeddingValues
    provenance: EmbeddingProvenance

    def __post_init__(self) -> None:
        """Validate that all variant embedding vectors align and are finite."""
        arrays = tuple(
            np.asarray(value, dtype=float).copy()
            for value in (
                self.reference,
                self.alternate,
                self.delta,
            )
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("variant embeddings must be one-dimensional")
        if len({array.shape for array in arrays}) != 1:
            raise ValueError("variant embedding vectors must have equal dimensions")
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("variant embedding values must be finite")
        for field, array in zip(("reference", "alternate", "delta"), arrays, strict=True):
            array.flags.writeable = False
            object.__setattr__(self, field, array)


class GenomeEmbedder(Protocol):
    """Backend-neutral contract for sequence representation models."""

    def embed(self, windows: Sequence[SequenceWindow]) -> EmbeddingBatch:
        """Encode sequence windows without interpreting them as treatment effects."""
        ...


class HFEncodingBackend(Protocol):
    """Testable low-level backend for batched model inference."""

    def encode(
        self,
        sequences: Sequence[str],
        config: HFGenomeEncoderConfig,
    ) -> EmbeddingValues:
        """Return one finite embedding per input sequence."""
        ...


class HFGenomeEmbedder:
    """Reproducible optional adapter for Hugging Face genome models.

    The adapter records immutable revisions and never exposes embeddings as
    biological-age or treatment-response evidence. Use a held-out
    :class:`~rejuvenationkit.genomics.calibration.GenomicTargetCalibrator`
    before converting an embedding-derived prediction to Phase 2 evidence.
    """

    def __init__(
        self,
        config: HFGenomeEncoderConfig,
        *,
        backend: HFEncodingBackend | None = None,
    ) -> None:
        """Initialize with an optional injected backend for offline testing."""
        self.config = config
        self._backend = backend or _TransformersEncodingBackend()

    def embed(self, windows: Sequence[SequenceWindow]) -> EmbeddingBatch:
        """Encode windows with explicit chunk and strand aggregation."""
        if not windows:
            raise ValueError("at least one sequence window is required")
        identifiers = tuple(item.sequence_id for item in windows)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("sequence window identifiers must be unique")
        expanded: list[str] = []
        ownership: list[int] = []
        aggregation_weights: list[float] = []
        for index, window in enumerate(windows):
            orientations = (
                (window.sequence, _reverse_complement(window.sequence))
                if self.config.strand_policy is StrandPolicy.FORWARD_REVERSE_MEAN
                else (window.sequence,)
            )
            chunks = tuple(
                weighted_chunk
                for orientation in orientations
                for weighted_chunk in self._weighted_chunks(orientation)
            )
            expanded.extend(chunk for chunk, _ in chunks)
            ownership.extend([index] * len(chunks))
            aggregation_weights.extend(weight for _, weight in chunks)
        encoded_parts: list[EmbeddingValues] = []
        for start in range(0, len(expanded), self.config.batch_size):
            batch = expanded[start : start + self.config.batch_size]
            encoded_parts.append(np.asarray(self._backend.encode(batch, self.config), dtype=float))
        encoded = np.vstack(encoded_parts)
        if encoded.shape[0] != len(expanded) or encoded.ndim != 2:
            raise ValueError("Hugging Face backend returned an invalid embedding shape")
        if not np.isfinite(encoded).all():
            raise ValueError("Hugging Face backend returned non-finite embeddings")
        ownership_array = np.asarray(ownership, dtype=int)
        weights_array = np.asarray(aggregation_weights, dtype=float)
        sums = np.zeros((len(windows), encoded.shape[1]), dtype=float)
        np.add.at(sums, ownership_array, encoded * weights_array[:, None])
        total_weights = np.bincount(
            ownership_array,
            weights=weights_array,
            minlength=len(windows),
        ).reshape(-1, 1)
        pooled = sums / total_weights
        provenance = EmbeddingProvenance(
            model_id=self.config.model_id,
            model_revision=self.config.model_revision,
            tokenizer_revision=self.config.tokenizer_revision or self.config.model_revision,
            weights_license=self.config.weights_license,
            layer=self.config.layer,
            model_class=self.config.model_class,
            pooling=self.config.pooling,
            strand_policy=self.config.strand_policy,
            overlength_policy=self.config.overlength_policy,
            maximum_bases=self.config.maximum_bases,
            chunk_overlap_bases=self.config.chunk_overlap_bases,
            maximum_tokens=self.config.maximum_tokens,
            batch_size=self.config.batch_size,
            device=self.config.device,
            trust_remote_code=self.config.trust_remote_code,
            local_files_only=self.config.local_files_only,
            library_versions=_installed_versions(),
            input_hashes={
                item.sequence_id: sha256(item.sequence.encode()).hexdigest() for item in windows
            },
        )
        return EmbeddingBatch(sequence_ids=identifiers, values=pooled, provenance=provenance)

    def embed_variant(
        self,
        variant_id: str,
        reference_window: SequenceWindow,
        alternate_window: SequenceWindow,
    ) -> VariantEmbedding:
        """Encode aligned SNV contexts and return alternate-minus-reference delta."""
        if len(reference_window.sequence) != len(alternate_window.sequence):
            raise ValueError("initial variant embedding supports equal-length allele contexts only")
        if reference_window.sequence_id == alternate_window.sequence_id:
            raise ValueError("reference and alternate windows need distinct sequence_ids")
        reference_coordinates = (
            reference_window.genome_assembly,
            reference_window.chromosome,
            reference_window.start,
            reference_window.end,
        )
        alternate_coordinates = (
            alternate_window.genome_assembly,
            alternate_window.chromosome,
            alternate_window.start,
            alternate_window.end,
        )
        if any(
            value is not None for value in (*reference_coordinates, *alternate_coordinates)
        ) and (reference_coordinates != alternate_coordinates):
            raise ValueError("reference and alternate windows must use identical coordinates")
        differing = [
            index
            for index, (reference, alternate) in enumerate(
                zip(reference_window.sequence, alternate_window.sequence, strict=True)
            )
            if reference != alternate
        ]
        if len(differing) != 1:
            raise ValueError("variant embedding requires exactly one SNV difference per context")
        reference_metadata = (
            reference_window.reference_allele,
            reference_window.alternate_allele,
        )
        alternate_metadata = (
            alternate_window.reference_allele,
            alternate_window.alternate_allele,
        )
        if (
            all(value is not None for value in reference_metadata)
            and all(value is not None for value in alternate_metadata)
            and reference_metadata != alternate_metadata
        ):
            raise ValueError("reference and alternate windows disagree on allele metadata")
        allele_metadata = (
            reference_metadata
            if reference_window.reference_allele is not None
            else alternate_metadata
        )
        if allele_metadata[0] is not None:
            if len(differing) != 1:
                raise ValueError("variant allele metadata requires exactly one sequence difference")
            difference = differing[0]
            reference_allele, alternate_allele = allele_metadata
            if len(reference_allele or "") != 1 or len(alternate_allele or "") != 1:
                raise ValueError("initial variant embedding supports single-base SNV alleles only")
            if reference_window.sequence[difference] != reference_allele:
                raise ValueError("reference allele does not match the reference sequence context")
            if alternate_window.sequence[difference] != alternate_allele:
                raise ValueError("alternate allele does not match the alternate sequence context")
        batch = self.embed((reference_window, alternate_window))
        reference = batch.values[0]
        alternate = batch.values[1]
        return VariantEmbedding(
            variant_id=variant_id,
            reference=reference,
            alternate=alternate,
            delta=alternate - reference,
            provenance=batch.provenance,
        )

    def _weighted_chunks(self, sequence: str) -> tuple[tuple[str, float], ...]:
        if len(sequence) <= self.config.maximum_bases:
            return ((sequence, float(len(sequence))),)
        if self.config.overlength_policy is OverlengthPolicy.ERROR:
            raise ValueError(
                f"sequence has {len(sequence)} bases, exceeding maximum_bases="
                f"{self.config.maximum_bases}; explicit chunking is required"
            )
        step = self.config.maximum_bases - self.config.chunk_overlap_bases
        chunks: list[tuple[str, float]] = []
        covered_until = 0
        for start in range(0, len(sequence), step):
            chunk = sequence[start : start + self.config.maximum_bases]
            if not chunk:
                continue
            end = start + len(chunk)
            newly_covered = end - max(start, covered_until)
            if newly_covered > 0:
                chunks.append((chunk, float(newly_covered)))
                covered_until = max(covered_until, end)
        return tuple(chunks)


class _TransformersEncodingBackend:
    """Lazy Transformers/PyTorch implementation used outside ordinary CI."""

    def __init__(self) -> None:
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def encode(
        self,
        sequences: Sequence[str],
        config: HFGenomeEncoderConfig,
    ) -> EmbeddingValues:
        """Run masked token pooling with padding and special tokens excluded."""
        self._load(config)
        if self._tokenizer is None or self._model is None or self._torch is None:
            raise RuntimeError("Hugging Face backend failed to load")
        encoded = self._tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
        )
        if int(encoded["attention_mask"].shape[1]) > config.maximum_tokens:
            raise ValueError(
                "tokenized input exceeds maximum_tokens; adjust base windows instead of truncating"
            )
        special_tokens_mask = encoded.pop("special_tokens_mask")
        model_inputs = {key: value.to(config.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            output = self._model(**model_inputs, output_hidden_states=True)
        hidden_states = output.hidden_states
        try:
            selected = hidden_states[config.layer]
        except IndexError as error:
            raise ValueError(f"requested model layer is unavailable: {config.layer}") from error
        attention = model_inputs["attention_mask"].bool()
        special = special_tokens_mask.to(config.device).bool()
        valid = attention & ~special
        denominator = valid.sum(dim=1, keepdim=True)
        if bool((denominator == 0).any()):
            raise ValueError("tokenizer produced no non-special sequence tokens")
        pooled = (selected * valid.unsqueeze(-1)).sum(dim=1) / denominator
        return np.asarray(pooled.detach().cpu().numpy(), dtype=float)

    def _load(self, config: HFGenomeEncoderConfig) -> None:
        if self._model is not None:
            return
        try:
            transformers = import_module("transformers")
            torch = import_module("torch")
        except ModuleNotFoundError as error:
            raise ImportError(
                "HFGenomeEmbedder requires optional dependencies; "
                "install rejuvenationkit[genome-hf]"
            ) from error
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.tokenizer_revision or config.model_revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        model_factory = (
            transformers.AutoModel
            if config.model_class is HFModelClass.AUTO_MODEL
            else transformers.AutoModelForMaskedLM
        )
        model = model_factory.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        model.to(config.device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {"rejuvenationkit-adapter": "1"}
    for package in ("transformers", "torch", "huggingface-hub"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "backend-injected-or-not-installed"
    return versions
