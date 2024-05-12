from unittest import TestCase

import numpy as np
import pandas as pd

from labcodes.fileio.base import DataDirectory, LogFile, LogName


class TestLogFile(TestCase):
    def test_init(self):
        df = pd.DataFrame(
            {
                "x": np.linspace(0, 10, 100),
                "y": np.random.normal(size=100),
            }
        )
        lf = LogFile(df, "x")
        self.assertEqual(lf.df.shape, (100, 2))
        self.assertEqual(lf.indeps, ["x"])
        self.assertEqual(lf.deps, ["y"])
        self.assertEqual(lf.meta, {})
        self.assertEqual(lf.name.id, 0)
        self.assertEqual(lf.name.title, "Untitled")
        self.assertIsNone(lf.save_path)


if __name__ == "__main__":
    import unittest

    unittest.main()
