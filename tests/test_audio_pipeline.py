import unittest

import numpy as np

from audio_pipeline import mix_tracks


class MixTracksTests(unittest.TestCase):
    def test_matches_converted_voice_level_and_keeps_stereo_shape(self):
        original_voice = np.full(1000, 0.2, dtype=np.float32)
        converted_voice = np.full(1000, 0.1, dtype=np.float32)
        accompaniment = np.zeros((2, 1000), dtype=np.float32)

        mixed = mix_tracks(converted_voice, original_voice, accompaniment)

        self.assertEqual(mixed.shape, (2, 1000))
        np.testing.assert_allclose(mixed, 0.2, atol=1e-6)

    def test_caps_output_peak_without_clipping(self):
        original_voice = np.full(100, 0.5, dtype=np.float32)
        converted_voice = np.full(100, 0.5, dtype=np.float32)
        accompaniment = np.full((2, 100), 0.8, dtype=np.float32)

        mixed = mix_tracks(converted_voice, original_voice, accompaniment)

        self.assertLessEqual(float(np.max(np.abs(mixed))), 0.98)

    def test_rejects_silent_reference(self):
        with self.assertRaisesRegex(ValueError, "silent"):
            mix_tracks(
                np.zeros(20, dtype=np.float32),
                np.ones(20, dtype=np.float32),
                np.zeros((2, 20), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
