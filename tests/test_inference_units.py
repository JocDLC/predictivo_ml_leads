"""Tests unitarios de helpers puros de core/inference.py (fixtures sintéticas, sin datos reales)."""
import unittest

from core import inference


class FixEncodingTests(unittest.TestCase):
    def test_repara_mojibake_latin1(self):
        self.assertEqual(inference.fix_encoding("MÃ©xico"), "México")

    def test_deja_intacto_texto_valido(self):
        self.assertEqual(inference.fix_encoding("Koleos"), "Koleos")


class ClasificarFranjaTests(unittest.TestCase):
    def test_madrugada(self):
        self.assertEqual(inference.clasificar_franja(3), "madrugada")

    def test_manana(self):
        self.assertEqual(inference.clasificar_franja(9), "manana")

    def test_tarde(self):
        self.assertEqual(inference.clasificar_franja(14), "tarde")

    def test_noche(self):
        self.assertEqual(inference.clasificar_franja(21), "noche")

    def test_bordes(self):
        self.assertEqual(inference.clasificar_franja(0), "madrugada")
        self.assertEqual(inference.clasificar_franja(6), "manana")
        self.assertEqual(inference.clasificar_franja(12), "tarde")
        self.assertEqual(inference.clasificar_franja(18), "noche")


class ResolveUmbralTests(unittest.TestCase):
    def test_umbral_legacy_se_reemplaza_por_default(self):
        self.assertEqual(inference.resolve_umbral({"umbral": inference.LEGACY_UMBRAL}), inference.DEFAULT_UMBRAL)

    def test_umbral_custom_se_respeta(self):
        self.assertEqual(inference.resolve_umbral({"umbral": 0.5}), 0.5)

    def test_sin_umbral_usa_default(self):
        self.assertEqual(inference.resolve_umbral({}), inference.DEFAULT_UMBRAL)

    def test_umbral_invalido_usa_default(self):
        self.assertEqual(inference.resolve_umbral({"umbral": "abc"}), inference.DEFAULT_UMBRAL)


class FindColumnTests(unittest.TestCase):
    def test_encuentra_columna_con_acentos(self):
        cols = ["Fecha de creación por el cliente", "Campaña"]
        self.assertEqual(inference.find_column(cols, ["creacion", "cliente"]), cols[0])

    def test_excluye_keywords(self):
        cols = ["Cualificación", "Sub-Cualificación"]
        self.assertEqual(
            inference.find_column(cols, ["cualificacion"], exclude_keywords=["sub"]),
            "Cualificación",
        )


if __name__ == "__main__":
    unittest.main()
