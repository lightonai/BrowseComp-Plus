"""Build a PLAID index for the BrowseComp-Plus corpus using PyLate ColBERT models."""

from __future__ import annotations

import argparse
import logging
import os

import torch
from datasets import load_dataset

from pylate import indexes, models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build a PyLate PLAID index for BrowseComp-Plus corpus"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="PyLate ColBERT model name or path (e.g. lightonai/GTE-ModernColBERT-v1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for the PLAID index (e.g. indexes/pylate/GTE-ModernColBERT-v1)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="Tevatron/browsecomp-plus-corpus",
        help="HuggingFace dataset name for the corpus",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for document encoding",
    )
    parser.add_argument(
        "--document-length",
        type=int,
        default=300,
        help="Maximum document length in tokens",
    )
    parser.add_argument(
        "--query-length",
        type=int,
        default=32,
        help="Maximum query length in tokens",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=4,
        help="Number of bits for PLAID quantization (default: 4)",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model inference",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code when loading model",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    # Load corpus
    logger.info(f"Loading corpus from {args.dataset_name}")
    ds = load_dataset(args.dataset_name, split="train")
    doc_ids = [row["docid"] for row in ds]
    doc_texts = [row["text"] for row in ds]
    logger.info(f"Loaded {len(doc_ids)} documents")

    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = models.ColBERT(
        model_name_or_path=args.model_name,
        document_length=args.document_length,
        query_length=args.query_length,
        model_kwargs={"torch_dtype": dtype_map[args.torch_dtype]},
        trust_remote_code=args.trust_remote_code,
    )

    # Encode documents
    logger.info(f"Encoding {len(doc_texts)} documents (batch_size={args.batch_size})")
    documents_embeddings = model.encode(
        sentences=doc_texts,
        batch_size=args.batch_size,
        is_query=False,
        show_progress_bar=True,
    )
    logger.info("Document encoding complete")

    # Get embedding size from first embedding
    embedding_size = documents_embeddings[0].shape[-1]
    logger.info(f"Embedding size: {embedding_size}")

    # Create index directory structure
    index_folder = os.path.dirname(args.output_dir.rstrip("/"))
    index_name = os.path.basename(args.output_dir.rstrip("/"))
    if not index_folder:
        index_folder = "."

    # Build PLAID index
    logger.info(f"Building PLAID index at {args.output_dir}")
    index = indexes.PLAID(
        index_folder=index_folder,
        index_name=index_name,
        override=True,
        nbits=args.nbits,
        embedding_size=embedding_size,
    )

    index.add_documents(
        documents_ids=doc_ids,
        documents_embeddings=documents_embeddings,
    )

    logger.info(f"Index built successfully at {args.output_dir}")


if __name__ == "__main__":
    main()
