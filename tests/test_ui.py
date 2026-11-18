from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class UiSmokeTests(unittest.TestCase):
    def test_app_starts_without_exception(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.button_group[0].value, "Analyze")
        self.assertGreaterEqual(len(app.button), 1)

    def test_every_navigation_page_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest

        for page in ("Results", "About"):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            app.button_group[0].set_value([page]).run()
            self.assertEqual(len(app.exception), 0, page)

    def test_ui_uses_lazy_page_router_not_tabs(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("st.segmented_control", source)
        self.assertNotIn("st.tabs(", source)
        self.assertIn('if page == "Analyze"', source)

    def test_ui_never_trains_on_launch(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn(".fit(", source)
        self.assertNotIn("train_bilstm", source)

    def test_ui_avoids_old_dashboard_clutter(self):
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("with st.sidebar", source)
        self.assertNotIn("examples =", source)
        self.assertNotIn("nearest_examples", source)
        self.assertNotIn('page == "Error lab"', source)


if __name__ == "__main__":
    unittest.main()
