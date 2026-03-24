#!/usr/bin/env python3
"""Check MCP call failures and token usage in a run output directory."""

import argparse
import json
import glob
import os


def check_failures(output_dir):
    pattern = os.path.join(output_dir, "run_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No run files found in {output_dir}")
        return

    print(f"Total files: {len(files)}")

    total_tokens = 0
    total_search_calls = 0
    failed_queries = []
    no_search_queries = []
    no_docids_queries = []
    high_token_queries = []

    for f in files:
        with open(f) as fh:
            d = json.load(fh)

        qid = d.get("query_id")
        status = d.get("status", "unknown")
        usage = d.get("usage", {})
        tokens = usage.get("total_tokens", 0)
        tc = d.get("tool_call_counts", {})
        search_calls = tc.get("search", 0)
        docids = d.get("retrieved_docids", [])

        total_tokens += tokens
        total_search_calls += search_calls

        # Check for failed MCP calls (output is None)
        has_failed = False
        for r in d.get("result", []):
            if r.get("type") == "tool_call" and r.get("output") is None:
                has_failed = True
                break

        if has_failed:
            failed_queries.append((qid, tokens, search_calls, status))

        if "search" not in tc:
            no_search_queries.append((qid, status))

        if not docids:
            no_docids_queries.append((qid, status, tc))

        if tokens > 50000:
            high_token_queries.append((qid, tokens, search_calls))

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries:        {len(files)}")
    print(f"Total tokens:         {total_tokens:,}")
    print(f"Avg tokens/query:     {total_tokens / len(files):,.0f}")
    print(f"Total search calls:   {total_search_calls}")
    print(f"Avg search calls:     {total_search_calls / len(files):.1f}")

    # Failed MCP calls
    print(f"\n{'='*60}")
    print(f"FAILED MCP CALLS: {len(failed_queries)} / {len(files)} ({100 * len(failed_queries) / len(files):.1f}%)")
    print(f"{'='*60}")
    if failed_queries:
        # Compare token usage
        failed_tokens = [t for _, t, _, _ in failed_queries]
        clean_tokens = [0] * (len(files) - len(failed_queries))
        clean_search = [0] * (len(files) - len(failed_queries))
        idx = 0
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            qid = d.get("query_id")
            if qid not in [q[0] for q in failed_queries]:
                usage = d.get("usage", {})
                clean_tokens[idx] = usage.get("total_tokens", 0)
                clean_search[idx] = d.get("tool_call_counts", {}).get("search", 0)
                idx += 1
                if idx >= len(clean_tokens):
                    break

        avg_failed_tokens = sum(t for _, t, _, _ in failed_queries) / len(failed_queries)
        avg_clean_tokens = sum(clean_tokens) / max(len(clean_tokens), 1)
        avg_failed_search = sum(s for _, _, s, _ in failed_queries) / len(failed_queries)
        avg_clean_search = sum(clean_search) / max(len(clean_search), 1)

        print(f"  Avg tokens (failed):  {avg_failed_tokens:,.0f}")
        print(f"  Avg tokens (clean):   {avg_clean_tokens:,.0f}")
        print(f"  Avg search (failed):  {avg_failed_search:.1f}")
        print(f"  Avg search (clean):   {avg_clean_search:.1f}")
        print()
        for qid, tokens, sc, status in failed_queries:
            print(f"  qid={qid:>6}  tokens={tokens:>8,}  search_calls={sc:>3}  status={status}")
    else:
        print("  None!")

    # No search tool calls
    if no_search_queries:
        print(f"\n{'='*60}")
        print(f"NO SEARCH TOOL CALLS: {len(no_search_queries)}")
        print(f"{'='*60}")
        for qid, status in no_search_queries:
            print(f"  qid={qid}  status={status}")

    # No retrieved docids
    if no_docids_queries:
        print(f"\n{'='*60}")
        print(f"NO RETRIEVED DOCIDS: {len(no_docids_queries)}")
        print(f"{'='*60}")
        for qid, status, tc in no_docids_queries:
            print(f"  qid={qid}  status={status}  tools={tc}")

    # High token queries
    print(f"\n{'='*60}")
    print(f"HIGH TOKEN QUERIES (>50K): {len(high_token_queries)} / {len(files)}")
    print(f"{'='*60}")
    for qid, tokens, sc in sorted(high_token_queries, key=lambda x: -x[1])[:20]:
        failed_marker = " [FAILED MCP]" if qid in [q[0] for q in failed_queries] else ""
        print(f"  qid={qid:>6}  tokens={tokens:>8,}  search_calls={sc:>3}{failed_marker}")
    if len(high_token_queries) > 20:
        print(f"  ... and {len(high_token_queries) - 20} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check MCP failures in run output directory")
    parser.add_argument(
        "output_dir",
        help="Path to the run output directory containing run_*.json files",
    )
    args = parser.parse_args()
    check_failures(args.output_dir)
