"""The licence's use grant names ORBTR's Terms of Service by its address, and the package ships the licence."""

import re
import tomllib

from helpers import ROOT


def test_the_use_grant_links_the_terms_and_the_package_ships_the_licence() -> None:
    license_text = (ROOT / "LICENSE").read_text()
    assert re.search(r"USE GRANT\.[\s\S]*Terms of Service \(https://orbtr\.ai/terms\)", license_text)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license-files"] == ["LICENSE"]
