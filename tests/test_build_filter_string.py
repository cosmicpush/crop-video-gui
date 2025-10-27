import unittest

from crop_gui import InteractiveVideoCropper


class BuildFilterStringTests(unittest.TestCase):
    def _make_cropper(self, target_res=None):
        return InteractiveVideoCropper(
            input_path="in.mp4",
            output_path="out.mp4",
            frame_ts=None,
            target_res=target_res,
            progress_callback=None,
            use_gpu=False,
        )

    def test_basic_crop_ensures_even_dimensions(self):
        cropper = self._make_cropper()

        result = cropper.build_filter_string(x=5, y=10, w=101, h=55)

        self.assertEqual(result, "crop=100:54:5:10")

    def test_target_resolution_adds_scale_and_crop_filters(self):
        cropper = self._make_cropper(target_res="1920x1080")

        result = cropper.build_filter_string(x=0, y=0, w=640, h=480)

        self.assertEqual(
            result,
            "crop=640:480:0:0,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
        )

    def test_invalid_target_resolution_is_ignored(self):
        cropper = self._make_cropper(target_res="4k")

        result = cropper.build_filter_string(x=0, y=0, w=640, h=480)

        self.assertEqual(result, "crop=640:480:0:0")


if __name__ == "__main__":
    unittest.main()
