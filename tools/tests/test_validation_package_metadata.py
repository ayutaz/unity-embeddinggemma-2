from collections import defaultdict
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'validation/com.ayutaz.embeddinggemma.validation'


def test_opt_in_validation_scripts_do_not_steal_existing_asset_guids():
    owners = defaultdict(list)
    for folder in ('Assets', 'Packages', 'validation'):
        for path in (ROOT / folder).rglob('*.meta'):
            match = re.search(r'^guid: ([a-f0-9]{32})$', path.read_text(encoding='utf-8'), re.M)
            assert match is not None, str(path.relative_to(ROOT))
            owners[match[1]].append(path.relative_to(ROOT).as_posix())
    collisions = [paths for paths in owners.values()
                  if len(paths) > 1 and any(p.startswith('validation/') for p in paths)]
    assert not collisions, collisions


def test_immutable_validation_package_has_metadata_for_every_imported_asset():
    missing = [path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob('*')
               if path.suffix != '.meta' and not Path(str(path) + '.meta').is_file()]
    assert not missing, missing
