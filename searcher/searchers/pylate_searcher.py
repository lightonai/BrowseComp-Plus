"""PyLate ColBERT searcher implementation using PLAID index."""

import logging
import os
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset

from pylate import indexes, models, retrieve

from .base import BaseSearcher

logger = logging.getLogger(__name__)


class PylateSearcher(BaseSearcher):
    @classmethod
    def parse_args(cls, parser):
        parser.add_argument(
            "--index-path",
            required=True,
            help="Path to the pre-built PLAID index directory (e.g. indexes/pylate/GTE-ModernColBERT-v1)",
        )
        parser.add_argument(
            "--model-name",
            required=True,
            help="PyLate ColBERT model name or path (e.g. lightonai/GTE-ModernColBERT-v1)",
        )
        parser.add_argument(
            "--dataset-name",
            default="Tevatron/browsecomp-plus-corpus",
            help="HuggingFace dataset for document text lookup (default: Tevatron/browsecomp-plus-corpus)",
        )
        parser.add_argument(
            "--document-length",
            type=int,
            default=300,
            help="Maximum document length in tokens (default: 300)",
        )
        parser.add_argument(
            "--query-length",
            type=int,
            default=32,
            help="Maximum query length in tokens (default: 32)",
        )
        parser.add_argument(
            "--torch-dtype",
            default="bfloat16",
            choices=["float16", "bfloat16", "float32"],
            help="Torch dtype for model inference (default: bfloat16)",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            default=False,
            help="Trust remote code when loading model",
        )

    def __init__(self, args):
        self.args = args

        logger.info("Initializing PyLate searcher...")

        self._load_model()
        self._load_index()
        self._load_dataset()

        logger.info("PyLate searcher initialized successfully")

    def _load_model(self) -> None:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        logger.info(f"Loading ColBERT model: {self.args.model_name}")
        self.model = models.ColBERT(
            model_name_or_path=self.args.model_name,
            document_length=self.args.document_length,
            query_length=self.args.query_length,
            model_kwargs={"torch_dtype": dtype_map[self.args.torch_dtype]},
            trust_remote_code=self.args.trust_remote_code,
        )
        logger.info("ColBERT model loaded successfully")

    def _load_index(self) -> None:
        index_path = self.args.index_path.rstrip("/")
        index_folder = os.path.dirname(index_path)
        index_name = os.path.basename(index_path)
        if not index_folder:
            index_folder = "."

        logger.info(f"Loading PLAID index from {self.args.index_path}")
        self.index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
            # batch_size=1<<8,
            batch_size=1<<10,
        )
        self.retriever = retrieve.ColBERT(index=self.index)
        logger.info("PLAID index loaded successfully")

    def _load_dataset(self) -> None:
        logger.info(f"Loading dataset: {self.args.dataset_name}")
        cache_dir = os.getenv("HF_DATASETS_CACHE")
        ds = load_dataset(self.args.dataset_name, split="train", cache_dir=cache_dir)
        self.docid_to_text = {row["docid"]: row["text"] for row in ds}
        logger.info(f"Loaded {len(self.docid_to_text)} documents for text lookup")

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        # Encode query
        query_embedding = self.model.encode(
            sentences=[query],
            is_query=True,
        )

        # Retrieve from PLAID index
        results = self.retriever.retrieve(
            queries_embeddings=query_embedding,
            k=k,
        )

        # Format results
        formatted = []
        for hit in results[0]:
            docid = hit["id"]
            formatted.append(
                {
                    "docid": docid,
                    "score": float(hit["score"]),
                    "text": self.docid_to_text.get(docid, "Text not found"),
                }
            )

        return formatted

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        text = self.docid_to_text.get(docid)
        if text is None:
            return None
        return {"docid": docid, "text": text}

    @property
    def search_type(self) -> str:
        return "PyLate-ColBERT"

    def search_description(self, k: int = 10) -> str:
        return f"Perform a ColBERT multi-vector search on the knowledge source. Returns top-{k} hits with docid, score, and snippet. The snippet contains the document's contents (may be truncated based on token limits)."

    def get_document_description(self) -> str:
        return "Retrieve a full document by its docid."
