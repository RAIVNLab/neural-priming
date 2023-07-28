from concurrent import futures
import time
import sqlite3
from pathlib import Path
import argparse
import subprocess
import os
import importlib
import sys


import pyarrow as pa
import pyarrow.parquet as pq
import itertools
import pandas as pd
import tqdm


current_directory = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(current_directory)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dbs", nargs="+", default=None, help="Path to sqlite dbs")
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        default=None,
        help="Pipe separated list of queries or newline separated query document",
    )

    parser.add_argument(
        "--template",
         action='store_true',
        help="",
    )
    parser.add_argument(
        "-n",
        "--quantity",
        type=int,
        default=None,
        help="Number of desired outputs (currently only functions with workers=1)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./",
        help="Full output path, will make parent directories if they don't exist",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--field", default="TEXT", type=str, help="Field to search database"
    )

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    if args.template:
        print('Priming.Templates.' + args.query)
        dataset_obj = importlib.import_module('Priming.Templates.' + args.query)
        words = dataset_obj.classes
    else:
        if not os.path.exists(args.query):
            words = args.query.split("|")
        else:
            words = [l for l in Path(args.query).read_text().split("\n") if l]

    print(
        f"Searching {len(args.dbs)} dbs for {len(words)} needles:"
    )
    out = search_sharded_database(
        args.dbs,
        words,
        workers=args.workers,
        max_results=args.quantity,
        field=args.field,
    )

    fields = [
        "SAMPLE_ID",
        "URL",
        "TEXT",
        "HEIGHT",
        "WIDTH",
        "LICENSE",
        "NSFW",
        "similarity",
        "QUERY",
    ]

    field_types = [
        pa.int64(),
        pa.binary(),
        pa.binary(),
        pa.int32(),
        pa.int32(),
        pa.binary(),
        pa.binary(),
        pa.float64(),
        pa.binary(),
    ]

    schema = pa.schema(
        [pa.field(name, dtype) for name, dtype in zip(fields, field_types)]
    )

    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(
        chunk_iterator(row_iterator(out, fn=process_fields), chunk_size=500000)
    ):
        df = pd.DataFrame(chunk, columns=fields)
        df.to_json(folder / f"chunk_{i}.json", orient="records")
        # table = pa.Table.from_pandas(df, schema=schema)
        # pq.write_table(table, folder / f"chunk_{i}.parquet")


def process_fields(key, row):
    sample_id, url, text, height, width, licence_, nsfw, similarity = row

    return (
        int(float(sample_id)) if sample_id else None,
        bytes(url, "utf-8") if url else None,
        bytes(text, "utf-8") if text else None,
        int(float(height)) if height else None,
        int(float(width)) if width else None,
        bytes(licence_, "utf-8") if licence_ else None,
        bytes(nsfw, "utf-8") if nsfw else None,
        float(similarity) if similarity else None,
        bytes(key, "utf-8"),
    )


def chunk_iterator(iterator, chunk_size):
    """
    Given an iterator, returns an iterator of iterators where each
    inner iterator has length `chunk_size` or less.
    """
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def row_iterator(in_dict, fn=lambda x: x):
    for key, values in in_dict.items():
        for row in values:
            yield fn(key, row)


def safe_dict_collate(dict_a, dict_b):
    set_keys = set(dict_a.keys()).union(set(dict_b.keys()))

    out = {}
    for k in set_keys:
        a_vals = dict_a.get(k, [])
        b_vals = dict_b.get(k, [])

        out[k] = a_vals + b_vals

    return out


def search_sharded_database(
    dbs, words, max_results=None, workers=1, field="TEXT"
):
    items = [(i, db, words, field) for i, db in enumerate(dbs)]

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures_to_results = {
            executor.submit(search_database, item): item for item in items
        }
        all_results = {}
        for future in futures.as_completed(futures_to_results):
            result = future.result()

            all_results = safe_dict_collate(all_results, result)

            if max_results is not None and all(
                [len(v) > max_results for v in all_results.values()]
            ):
                for future in futures_to_results:
                    future.cancel()
                break

    return all_results


def search_database(args):
    shard_idx, db, words, field = args
    word_to_results = {}
    start_time = time.time()
    total_results = 0
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        c = conn.cursor()
        for i, word in tqdm.tqdm(enumerate(words), desc=f"Shard {shard_idx} ", total=len(words)):
            query = f"SELECT * FROM samples WHERE {field} MATCH '\"{word}\"'"
            c.execute(query)

            # Fetch results
            word_to_results[word] = list(c.fetchall())
            total_results += len(word_to_results[word])

        end_time = time.time()
        print(
            f"Search of shard {shard_idx} took {end_time - start_time:.4f} seconds for {len(words)} words,"
            f" {total_results} results"
        )

        conn.close()
    else:
        print("Skipping shard:{}".format(db))
    return word_to_results


if __name__ == "__main__":
    main()
