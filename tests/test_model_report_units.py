"""Tests del degradado elegante del reporte cuando faltan datasets locales."""
import tempfile
import unittest
from unittest import mock

from components import model_report


class TryLoadReportArtifactsTests(unittest.TestCase):
    def test_devuelve_none_si_falta_dataset_train_test(self):
        fake_model = mock.Mock()
        fake_model.feature_names_in_ = ["f1", "f2"]
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch.object(model_report, "DATA_DIR", tmpdir),
            mock.patch.object(
                model_report,
                "load_model_and_artifacts",
                return_value=(fake_model, {"umbral": 0.4}),
            ),
        ):
            self.assertIsNone(model_report.try_load_report_artifacts())


if __name__ == "__main__":
    unittest.main()
