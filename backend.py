# -*- coding: utf-8 -*-
from bw2data import mapping, databases, config
from bw2data.backends.peewee import SQLiteBackend
from bw2data.utils import MAX_INT_32, TYPE_DICTIONARY
from bw2data.errors import UnknownObject
import datetime
import itertools
import numpy as np


class ProcessedBackend(SQLiteBackend):
    """Don't write the data, just the processed arrays."""
    backend = "processed"

    def write(self, data):
        """Process inventory documents to NumPy structured arrays."""
        if self.name not in databases:
            self.register()

        mapping.add(data.keys())

        # Figure out when the production exchanges are implicit
        missing_production_keys = [
            key for key, value in data.items()
            if not [1 for exc in value.get('exchanges') if exc['type'] == 'production']
        ]

        num_exchanges = sum(1 for ds in data.values() for exc in ds.get('exchanges'))
        num_processes = len(data)

        arr = np.zeros((num_exchanges + len(missing_production_keys), ), dtype=self.dtype)

        for index, row in enumerate(exc for ds in data.values() for exc in ds.get('exchanges')):
            if "type" not in row:
                raise UntypedExchange
            if "amount" not in row or "input" not in row:
                raise InvalidExchange
            if np.isnan(row['amount']) or np.isinf(row['amount']):
                raise ValueError("Invalid amount in exchange {}".format(row))

            try:
                arr[index] = (
                    mapping[row['input']],
                    mapping[row['output']],
                    MAX_INT_32,
                    MAX_INT_32,
                    TYPE_DICTIONARY[row["type"]],
                    row.get("uncertainty type", 0),
                    row["amount"],
                    row["amount"] \
                        if row.get("uncertainty type", 0) in (0,1) \
                        else row.get("loc", np.NaN),
                    row.get("scale", np.NaN),
                    row.get("shape", np.NaN),
                    row.get("minimum", np.NaN),
                    row.get("maximum", np.NaN),
                    row["amount"] < 0
                )
            except KeyError:
                raise UnknownObject(("Exchange between {} and {} is invalid "
                    "- one of these objects is unknown (i.e. doesn't exist "
                    "as a process dataset)"
                    ).format(
                        row['input'],
                        row['output']
                    )
                )

        # If exchanges were found, start inserting rows at len(exchanges) + 1
        index += 1

        for index, key in zip(itertools.count(index), missing_production_keys):
            arr[index] = (
                mapping[key], mapping[key],
                MAX_INT_32, MAX_INT_32, TYPE_DICTIONARY["production"],
                0, 1, 1, np.NaN, np.NaN, np.NaN, np.NaN, False
            )

        databases[self.name]['depends'] = ['biosphere3', 'ecoinvent']  # Why are we hard-coding this again!?
        databases[self.name]['processed'] = datetime.datetime.now().isoformat()
        databases.flush()

        arr.sort(order=self.dtype_field_order())
        np.save(self.filepath_processed(), arr, allow_pickle=False)

    def process(self):
        """No-op; no intermediate data to process"""
        return


config.backends['processed'] = ProcessedBackend
