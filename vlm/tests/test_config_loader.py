from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config_loader import ConfigValidationError, load_config


def valid_config() -> dict:
    return {
        "environment": "test",
        "device": "cpu",
        "max_new_tokens": 60,
        "use_mock_model": True,
        "model_id": "test/model",
        "image_longest_edge": 1024,
        "max_image_size": 512,
        "extra_key": "allowed",
    }


class ConfigLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "config.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write(self, value: object) -> None:
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def test_loads_valid_config_and_allows_extra_keys(self) -> None:
        config = valid_config()
        self.write(config)
        self.assertEqual(load_config(self.path), config)

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config(self.path)

    def test_invalid_json(self) -> None:
        self.path.write_text("{invalid", encoding="utf-8")
        with self.assertRaisesRegex(ConfigValidationError, "JSON"):
            load_config(self.path)

    def test_top_level_list(self) -> None:
        self.write([])
        with self.assertRaisesRegex(ConfigValidationError, "최상위"):
            load_config(self.path)

    def test_missing_required_key(self) -> None:
        config = valid_config()
        del config["model_id"]
        self.write(config)
        with self.assertRaisesRegex(ConfigValidationError, "model_id"):
            load_config(self.path)

    def test_invalid_device(self) -> None:
        config = valid_config()
        config["device"] = "tpu"
        self.write(config)
        with self.assertRaisesRegex(ConfigValidationError, "device"):
            load_config(self.path)

    def test_bool_type_error(self) -> None:
        config = valid_config()
        config["use_mock_model"] = 1
        self.write(config)
        with self.assertRaisesRegex(ConfigValidationError, "use_mock_model"):
            load_config(self.path)

    def test_unsupported_temperature_key_rejected(self) -> None:
        config = valid_config()
        config["temperature"] = 0.1
        self.write(config)
        with self.assertRaisesRegex(ConfigValidationError, "temperature"):
            load_config(self.path)

    def test_unimplemented_quantization_rejected(self) -> None:
        config = valid_config()
        config["quantization"] = "4bit"
        self.write(config)
        with self.assertRaisesRegex(ConfigValidationError, "quantization"):
            load_config(self.path)

    def test_quantization_none_allowed(self) -> None:
        config = valid_config()
        config["quantization"] = "none"
        self.write(config)
        self.assertEqual(load_config(self.path)["quantization"], "none")

    def test_shipped_configs_are_valid(self) -> None:
        for name in ("jetson.json", "pc.json"):
            with self.subTest(config=name):
                config = load_config(Path(__file__).resolve().parents[1] / "config" / name)
                self.assertNotIn("temperature", config)
                self.assertEqual(config.get("quantization"), "none")

    def test_positive_integer_errors(self) -> None:
        for key, value in (
            ("max_new_tokens", 0),
            ("image_longest_edge", -1),
            ("max_image_size", True),
        ):
            with self.subTest(key=key):
                config = valid_config()
                config[key] = value
                self.write(config)
                with self.assertRaisesRegex(ConfigValidationError, key):
                    load_config(self.path)


if __name__ == "__main__":
    unittest.main()
