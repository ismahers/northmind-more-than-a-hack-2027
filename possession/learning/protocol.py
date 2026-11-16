"""Fixed outer LOMO + one rotating inner validation match; never split windows randomly."""

from dataclasses import dataclass

from possession.external_data.idsse import IDSSE_IDS
from possession.external_data.schema import Match, file_hash


@dataclass(frozen=True)
class Fold:
    name: str
    train: tuple
    validation: tuple
    test: tuple

    def validate(self, keys):
        groups = [set(self.train), set(self.validation), set(self.test)]
        if not all(groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Nonempty disjoint match splits required")
        if set.union(*groups) != set(keys):
            raise ValueError("Fold does not partition supplied matches")
        if any(len(group) != len(set(group)) for group in (self.train, self.validation, self.test)):
            raise ValueError("Duplicate split keys")
        return self

    def to_dict(self):
        return {"name": self.name, "train": list(self.train),
                "validation": list(self.validation), "test": list(self.test)}


def lomo(keys, require_seven=True):
    keys = sorted(keys)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate match keys")
    idsse = [key for key in keys if key.startswith("idsse:")]
    expected = {"idsse:DFL-MAT-" + mid for mid in IDSSE_IDS}
    if require_seven and set(idsse) != expected:
        raise ValueError("Official IDSSE LOMO requires exactly the seven documented matches")
    if len(idsse) < 3 or any(not key.startswith(("idsse:", "skillcorner:")) for key in keys):
        raise ValueError("Need >=3 IDSSE matches and only supported public providers")
    folds = []
    for i, test in enumerate(idsse):
        validation = idsse[(i - 1) % len(idsse)]
        train = tuple(key for key in keys if key not in (test, validation))
        folds.append(Fold("idsse-" + str(i), train, (validation,), (test,)).validate(keys))
    return folds


def inventory(paths):
    """Hash artifacts and native sources; reject alias copies across differently named matches."""
    result, native_owner = {}, {}
    for path in paths:
        metadata = Match.read_metadata(path)
        key = metadata["provider"] + ":" + metadata["match_id"]
        if key in result:
            raise ValueError("Duplicate match identity")
        for source in metadata["sources"]:
            digest = source["sha256"]
            if digest in native_owner and native_owner[digest] != key:
                raise ValueError("Native source reused under multiple match identities")
            native_owner[digest] = key
        result[key] = {"path": str(path), "sha256": file_hash(path), "native_sources": metadata["sources"]}
    return result
