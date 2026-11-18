"""The sixteen meters (buḥūr), their forms and their ʿarūḍ / ḍarb tables, as machine-readable data.

Every pattern is *derived* from the base feet by the operations in feet.py, never written by hand, so a wrong table entry
shows up as a wrong operation name rather than a silent typo in a 0/1 string.

Structure of one form (e.g. البسيط المجزوء):
  hashw      the base feet of a hemistich in order, the last one being the foot the ʿarūḍ / ḍarb is built from
  hashw_zihaf  allowed zihāfāt per hashw position (the last position is governed by the end forms instead)
  aruds / darbs  named end forms: base foot + a binding zihāf/ʿilla (lāzim: kept in every bayt of the poem) + optional
               free zihāfāt (may vary from bayt to bayt)
  pairs      which ḍarb forms each ʿarūḍ form allows (classical tables)

Rarity: zihāfāt the books describe as permitted but rare or "qabīḥ" are flagged; the scanner prefers analyses without them
and reports them, but they are not violations.

Sources for the tables: the standard Khalīlian tables (16 meters, 36 ʿarūḍ, 63 ḍarb) as presented in the classical manuals
(e.g. al-Hāshimī, Mīzān al-Dhahab); cross-checked programmatically against two independent open-source implementations
(ARBML Bohour and pyarud). Every difference found by that cross-check is listed in validation/spec_crosscheck.md with the
decision taken.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .feet import FEET, ILLA_NAMES, ZIHAF_NAMES, ZIHAF_PARTS, InvalidOperation, apply_illa, apply_zihaf


def combine_zihaf(*keys: str | None) -> str | None:
    """Combine single/double zihāfāt into one key (khabn + ṭayy -> khabl). None if nothing applies."""
    parts = set()
    for k in keys:
        if k:
            parts |= set(ZIHAF_PARTS[k])
    if not parts:
        return None
    for k, p in ZIHAF_PARTS.items():
        if set(p) == parts:
            return k
    raise InvalidOperation(f"no zihāf combines {sorted(parts)}")


@dataclass(frozen=True)
class EndForm:
    """An ʿarūḍ or ḍarb form: base foot + lāzim zihāf / ʿilla + optional free zihāfāt."""
    key: str
    name: str                       # Arabic description, e.g. "مقطوعة"
    foot: str
    zihaf: str | None = None        # binding zihāf (part of the form)
    illa: str | None = None         # binding ʿilla
    free: tuple = ()                # zihāfāt that may additionally occur in some bayts
    rare_free: tuple = ()           # free zihāfāt that are permitted but rare

    def variants(self) -> list[dict]:
        """All realisations: [{pattern, zihaf, illa, rare}] (the lāzim part included)."""
        out, seen = [], set()
        for extra, rare in [(None, False)] + [(z, False) for z in self.free] + [(z, True) for z in self.rare_free]:
            try:
                z = combine_zihaf(self.zihaf, extra)
                foot = FEET[self.foot]
                if self.illa in ("qatf", "tasheeth"):
                    if z:
                        continue
                    p = apply_illa(foot, self.illa)
                else:
                    p = apply_zihaf(foot, z) if z else foot.pattern
                    if self.illa:
                        p = apply_illa(foot, self.illa, p)
            except InvalidOperation:
                continue
            if (p, z) in seen:
                continue
            seen.add((p, z))
            out.append({"pattern": p, "zihaf": z, "illa": self.illa, "rare": rare})
        return out


@dataclass(frozen=True)
class Form:
    key: str                        # tam / majzu / mukhalla / ahadh
    name: str
    hashw: tuple                    # base feet of one hemistich, last = end foot
    hashw_zihaf: tuple              # per hashw position (len = len(hashw) - 1): tuple of (zihaf, rare)
    aruds: tuple
    darbs: tuple
    pairs: dict                     # arud key -> tuple of darb keys
    muaqaba: tuple = ()             # forbidden adjacent (kaff on foot i, khabn on foot i+1) positions (i, i+1)
    muraqaba: tuple = ()            # hashw positions whose foot may not stay sound (المضارع، المقتضب)

    def end(self, key: str) -> EndForm:
        for e in self.aruds + self.darbs:
            if e.key == key:
                return e
        raise KeyError(key)


@dataclass(frozen=True)
class Meter:
    key: str
    name: str                       # Arabic, as used by the Shaer data and the BiLSTM classifier (with ال)
    forms: dict = field(default_factory=dict)


def _hz(*items) -> tuple:
    """hashw zihāf list: names, with a trailing '?' marking a rare one."""
    return tuple((i.rstrip("?"), i.endswith("?")) for i in items)


E = EndForm
METERS: dict[str, Meter] = {}


def _meter(key, name, *forms):
    METERS[key] = Meter(key, name, {f.key: f for f in forms})


# ---------------------------------------------------------------------------------------------- الطويل
_meter("tawil", "الطويل",
       Form("tam", "تام", ("faulun", "mafailun", "faulun", "mafailun"),
            (_hz("qabd"), _hz("qabd", "kaff?"), _hz("qabd")),
            aruds=(E("A_qabd", "مقبوضة", "mafailun", zihaf="qabd"),),
            darbs=(E("D_sahih", "صحيح", "mafailun"), E("D_qabd", "مقبوض", "mafailun", zihaf="qabd"),
                   E("D_hadhf", "محذوف", "mafailun", illa="hadhf")),
            pairs={"A_qabd": ("D_sahih", "D_qabd", "D_hadhf")}))

# ---------------------------------------------------------------------------------------------- المديد (always majzūʾ)
_meter("madid", "المديد",
       Form("tam", "تام", ("failatun", "failun", "failatun"),
            (_hz("khabn", "kaff", "shakl?"), _hz("khabn")),
            aruds=(E("A_sahih", "صحيحة", "failatun", free=("khabn",)),
                   E("A_hadhf", "محذوفة", "failatun", illa="hadhf"),
                   E("A_hadhf_khabn", "محذوفة مخبونة", "failatun", zihaf="khabn", illa="hadhf")),
            darbs=(E("D_sahih", "صحيح", "failatun", free=("khabn",)),
                   E("D_qasr", "مقصور", "failatun", illa="qasr", free=("khabn",)),
                   E("D_hadhf", "محذوف", "failatun", illa="hadhf"),
                   E("D_batr", "أبتر", "failatun", illa="batr"),
                   E("D_hadhf_khabn", "محذوف مخبون", "failatun", zihaf="khabn", illa="hadhf")),
            pairs={"A_sahih": ("D_sahih",), "A_hadhf": ("D_qasr", "D_hadhf", "D_batr"),
                   "A_hadhf_khabn": ("D_hadhf_khabn", "D_batr")},
            muaqaba=((0, 1),)))

# ---------------------------------------------------------------------------------------------- البسيط
_BASIT_H = (_hz("khabn", "tayy", "khabl?"), _hz("khabn"))
_meter("basit", "البسيط",
       Form("tam", "تام", ("mustafilun", "failun", "mustafilun", "failun"),
            _BASIT_H + (_hz("khabn", "tayy", "khabl?"),),
            aruds=(E("A_khabn", "مخبونة", "failun", zihaf="khabn"),),
            darbs=(E("D_khabn", "مخبون", "failun", zihaf="khabn"), E("D_qat", "مقطوع", "failun", illa="qat")),
            pairs={"A_khabn": ("D_khabn", "D_qat")}),
       Form("majzu", "مجزوء", ("mustafilun", "failun", "mustafilun"), _BASIT_H,
            aruds=(E("A_sahih", "صحيحة", "mustafilun", free=("khabn", "tayy")),
                   E("A_qat", "مقطوعة", "mustafilun", illa="qat")),
            darbs=(E("D_tadhyil", "مذال", "mustafilun", illa="tadhyil", free=("khabn", "tayy")),
                   E("D_sahih", "صحيح", "mustafilun", free=("khabn", "tayy")),
                   E("D_qat", "مقطوع", "mustafilun", illa="qat")),
            pairs={"A_sahih": ("D_tadhyil", "D_sahih", "D_qat"), "A_qat": ("D_qat",)}),
       Form("mukhalla", "مخلع", ("mustafilun", "failun", "mustafilun"), _BASIT_H,
            aruds=(E("A_khabn_qat", "مخلعة (مخبونة مقطوعة)", "mustafilun", zihaf="khabn", illa="qat"),),
            darbs=(E("D_khabn_qat", "مخلع (مخبون مقطوع)", "mustafilun", zihaf="khabn", illa="qat"),),
            pairs={"A_khabn_qat": ("D_khabn_qat",)}))

# ---------------------------------------------------------------------------------------------- الوافر
_WAFIR_H = _hz("asb", "aql?", "naqs?")
_meter("wafir", "الوافر",
       Form("tam", "تام", ("mufaalatun", "mufaalatun", "mufaalatun"), (_WAFIR_H, _WAFIR_H),
            aruds=(E("A_qatf", "مقطوفة", "mufaalatun", illa="qatf"),),
            darbs=(E("D_qatf", "مقطوف", "mufaalatun", illa="qatf"),),
            pairs={"A_qatf": ("D_qatf",)}),
       Form("majzu", "مجزوء", ("mufaalatun", "mufaalatun"), (_WAFIR_H,),
            aruds=(E("A_sahih", "صحيحة", "mufaalatun", free=("asb",)),),
            darbs=(E("D_sahih", "صحيح", "mufaalatun", free=("asb",)),
                   E("D_asb", "معصوب", "mufaalatun", zihaf="asb")),
            pairs={"A_sahih": ("D_sahih", "D_asb")}))

# ---------------------------------------------------------------------------------------------- الكامل
_KAMIL_H = _hz("idmar", "waqs?", "khazl?")
_KAMIL_TAM_A = (E("A_sahih", "صحيحة", "mutafailun", free=("idmar",), rare_free=("waqs", "khazl")),
                E("A_hadhadh", "حذاء", "mutafailun", illa="hadhadh", rare_free=("idmar",)))
_KAMIL_TAM_D = (E("D_sahih", "صحيح", "mutafailun", free=("idmar",), rare_free=("waqs", "khazl")),
                E("D_qat", "مقطوع", "mutafailun", illa="qat", free=("idmar",)),
                E("D_hadhadh_idmar", "أحذ مضمر", "mutafailun", zihaf="idmar", illa="hadhadh"),
                E("D_hadhadh", "أحذ", "mutafailun", illa="hadhadh"))
_meter("kamil", "الكامل",
       Form("tam", "تام", ("mutafailun",) * 3, (_KAMIL_H, _KAMIL_H), aruds=_KAMIL_TAM_A, darbs=_KAMIL_TAM_D,
            pairs={"A_sahih": ("D_sahih", "D_qat", "D_hadhadh_idmar"), "A_hadhadh": ("D_hadhadh", "D_hadhadh_idmar")}),
       Form("ahadh", "أحذ", ("mutafailun",) * 3, (_KAMIL_H, _KAMIL_H), aruds=_KAMIL_TAM_A, darbs=_KAMIL_TAM_D,
            pairs={"A_sahih": ("D_hadhadh_idmar",), "A_hadhadh": ("D_hadhadh", "D_hadhadh_idmar")}),
       Form("majzu", "مجزوء", ("mutafailun",) * 2, (_KAMIL_H,),
            aruds=(E("A_sahih", "صحيحة", "mutafailun", free=("idmar",), rare_free=("waqs", "khazl")),),
            darbs=(E("D_tarfil", "مرفل", "mutafailun", illa="tarfil", free=("idmar",)),
                   E("D_tadhyil", "مذيل", "mutafailun", illa="tadhyil", free=("idmar",)),
                   E("D_sahih", "صحيح", "mutafailun", free=("idmar",), rare_free=("waqs", "khazl")),
                   E("D_qat", "مقطوع", "mutafailun", illa="qat", free=("idmar",))),
            pairs={"A_sahih": ("D_tarfil", "D_tadhyil", "D_sahih", "D_qat")}))

# ---------------------------------------------------------------------------------------------- الهزج (always majzūʾ)
_meter("hazaj", "الهزج",
       Form("tam", "تام", ("mafailun", "mafailun"), (_hz("kaff", "qabd?"),),
            aruds=(E("A_sahih", "صحيحة", "mafailun"),),
            darbs=(E("D_sahih", "صحيح", "mafailun"), E("D_hadhf", "محذوف", "mafailun", illa="hadhf")),
            pairs={"A_sahih": ("D_sahih", "D_hadhf")}))

# ---------------------------------------------------------------------------------------------- الرجز
_RAJAZ_H = _hz("khabn", "tayy", "khabl?")
_meter("rajaz", "الرجز",
       Form("tam", "تام", ("mustafilun",) * 3, (_RAJAZ_H, _RAJAZ_H),
            aruds=(E("A_sahih", "صحيحة", "mustafilun", free=("khabn", "tayy"), rare_free=("khabl",)),),
            darbs=(E("D_sahih", "صحيح", "mustafilun", free=("khabn", "tayy"), rare_free=("khabl",)),
                   E("D_qat", "مقطوع", "mustafilun", illa="qat", free=("khabn",))),
            pairs={"A_sahih": ("D_sahih", "D_qat")}),
       Form("majzu", "مجزوء", ("mustafilun",) * 2, (_RAJAZ_H,),
            aruds=(E("A_sahih", "صحيحة", "mustafilun", free=("khabn", "tayy"), rare_free=("khabl",)),),
            darbs=(E("D_sahih", "صحيح", "mustafilun", free=("khabn", "tayy"), rare_free=("khabl",)),),
            pairs={"A_sahih": ("D_sahih",)}))

# ---------------------------------------------------------------------------------------------- الرمل
_RAMAL_H = _hz("khabn", "kaff", "shakl?")
_meter("ramal", "الرمل",
       Form("tam", "تام", ("failatun",) * 3, (_RAMAL_H, _RAMAL_H),
            aruds=(E("A_hadhf", "محذوفة", "failatun", illa="hadhf", free=("khabn",)),),
            darbs=(E("D_sahih", "صحيح", "failatun", free=("khabn",)),
                   E("D_qasr", "مقصور", "failatun", illa="qasr", free=("khabn",)),
                   E("D_hadhf", "محذوف", "failatun", illa="hadhf", free=("khabn",))),
            pairs={"A_hadhf": ("D_sahih", "D_qasr", "D_hadhf")}, muaqaba=((0, 1), (1, 2))),
       Form("majzu", "مجزوء", ("failatun",) * 2, (_RAMAL_H,),
            aruds=(E("A_sahih", "صحيحة", "failatun", free=("khabn",)),),
            darbs=(E("D_tasbigh", "مسبغ", "failatun", illa="tasbigh", rare_free=("khabn",)),
                   E("D_sahih", "صحيح", "failatun", free=("khabn",)),
                   E("D_hadhf", "محذوف", "failatun", illa="hadhf", free=("khabn",))),
            pairs={"A_sahih": ("D_tasbigh", "D_sahih", "D_hadhf")}, muaqaba=((0, 1),)))

# ---------------------------------------------------------------------------------------------- السريع
_SARI_H = _hz("khabn", "tayy", "khabl?")
_meter("sari", "السريع",
       Form("tam", "تام", ("mustafilun", "mustafilun", "mafulatu"), (_SARI_H, _SARI_H),
            aruds=(E("A_tayy_kashf", "مطوية مكشوفة", "mafulatu", zihaf="tayy", illa="kashf"),
                   E("A_khabl_kashf", "مخبولة مكشوفة", "mafulatu", zihaf="khabl", illa="kashf")),
            darbs=(E("D_tayy_waqf", "مطوي موقوف", "mafulatu", zihaf="tayy", illa="waqf"),
                   E("D_tayy_kashf", "مطوي مكشوف", "mafulatu", zihaf="tayy", illa="kashf"),
                   E("D_salm", "أصلم", "mafulatu", illa="salm"),
                   E("D_khabl_kashf", "مخبول مكشوف", "mafulatu", zihaf="khabl", illa="kashf")),
            pairs={"A_tayy_kashf": ("D_tayy_waqf", "D_tayy_kashf", "D_salm"), "A_khabl_kashf": ("D_khabl_kashf", "D_salm")}))

# ---------------------------------------------------------------------------------------------- المنسرح
_meter("munsarih", "المنسرح",
       Form("tam", "تام", ("mustafilun", "mafulatu", "mustafilun"),
            (_hz("khabn", "tayy", "khabl?"), _hz("tayy", "khabn", "khabl?")),
            aruds=(E("A_sahih", "صحيحة", "mustafilun", free=("tayy", "khabn")),),
            darbs=(E("D_tayy", "مطوي", "mustafilun", zihaf="tayy"), E("D_qat", "مقطوع", "mustafilun", illa="qat")),
            pairs={"A_sahih": ("D_tayy", "D_qat")}))

# ---------------------------------------------------------------------------------------------- الخفيف
_meter("khafif", "الخفيف",
       Form("tam", "تام", ("failatun", "mustafi_lun", "failatun"),
            (_hz("khabn", "kaff?", "shakl?"), _hz("khabn", "kaff?", "shakl?")),
            aruds=(E("A_sahih", "صحيحة", "failatun", free=("khabn",)),
                   E("A_tasheeth", "مشعثة", "failatun", illa="tasheeth"),
                   E("A_hadhf", "محذوفة", "failatun", illa="hadhf", free=("khabn",))),
            darbs=(E("D_sahih", "صحيح", "failatun", free=("khabn",)),
                   E("D_tasheeth", "مشعث", "failatun", illa="tasheeth"),
                   E("D_hadhf", "محذوف", "failatun", illa="hadhf", free=("khabn",))),
            pairs={"A_sahih": ("D_sahih", "D_tasheeth", "D_hadhf"), "A_tasheeth": ("D_sahih", "D_tasheeth", "D_hadhf"),
                   "A_hadhf": ("D_hadhf",)},
            muaqaba=((0, 1), (1, 2))),
       Form("majzu", "مجزوء", ("failatun", "mustafi_lun"), (_hz("khabn", "kaff?", "shakl?"),),
            aruds=(E("A_sahih", "صحيحة", "mustafi_lun", free=("khabn",)),),
            darbs=(E("D_sahih", "صحيح", "mustafi_lun", free=("khabn",)),
                   E("D_khabn_qasr", "مخبون مقصور", "mustafi_lun", zihaf="khabn", illa="qasr")),
            pairs={"A_sahih": ("D_sahih", "D_khabn_qasr")}, muaqaba=((0, 1),)))

# ---------------------------------------------------------------------------------------------- المضارع (always majzūʾ)
_meter("mudari", "المضارع",
       Form("tam", "تام", ("mafailun", "fai_latun"), (_hz("qabd", "kaff"),),
            aruds=(E("A_sahih", "صحيحة", "fai_latun"),),
            darbs=(E("D_sahih", "صحيح", "fai_latun"),),
            pairs={"A_sahih": ("D_sahih",)}, muraqaba=(0,)))

# ---------------------------------------------------------------------------------------------- المقتضب (always majzūʾ)
_meter("muqtadab", "المقتضب",
       Form("tam", "تام", ("mafulatu", "mustafilun"), (_hz("tayy", "khabn"),),
            aruds=(E("A_tayy", "مطوية", "mustafilun", zihaf="tayy"),),
            darbs=(E("D_tayy", "مطوي", "mustafilun", zihaf="tayy"),),
            pairs={"A_tayy": ("D_tayy",)}, muraqaba=(0,)))

# ---------------------------------------------------------------------------------------------- المجتث (always majzūʾ)
_meter("mujtath", "المجتث",
       Form("tam", "تام", ("mustafi_lun", "failatun"), (_hz("khabn", "kaff?", "shakl?"),),
            aruds=(E("A_sahih", "صحيحة", "failatun", free=("khabn",)),
                   E("A_tasheeth", "مشعثة", "failatun", illa="tasheeth")),
            darbs=(E("D_sahih", "صحيح", "failatun", free=("khabn",)),
                   E("D_tasheeth", "مشعث", "failatun", illa="tasheeth")),
            pairs={"A_sahih": ("D_sahih", "D_tasheeth"), "A_tasheeth": ("D_sahih", "D_tasheeth")},
            muaqaba=((0, 1),)))

# ---------------------------------------------------------------------------------------------- المتقارب
_meter("mutaqarib", "المتقارب",
       Form("tam", "تام", ("faulun",) * 4, (_hz("qabd"),) * 3,
            aruds=(E("A_sahih", "صحيحة", "faulun"), E("A_hadhf", "محذوفة", "faulun", illa="hadhf")),
            darbs=(E("D_sahih", "صحيح", "faulun"), E("D_qasr", "مقصور", "faulun", illa="qasr"),
                   E("D_hadhf", "محذوف", "faulun", illa="hadhf"), E("D_batr", "أبتر", "faulun", illa="batr")),
            pairs={"A_sahih": ("D_sahih", "D_qasr", "D_hadhf", "D_batr"), "A_hadhf": ("D_hadhf", "D_batr")}),
       Form("majzu", "مجزوء", ("faulun",) * 3, (_hz("qabd"),) * 2,
            aruds=(E("A_hadhf", "محذوفة", "faulun", illa="hadhf"),),
            darbs=(E("D_hadhf", "محذوف", "faulun", illa="hadhf"), E("D_batr", "أبتر", "faulun", illa="batr")),
            pairs={"A_hadhf": ("D_hadhf", "D_batr")}))

# ---------------------------------------------------------------------------------------------- المتدارك
_meter("mutadarik", "المتدارك",
       Form("tam", "تام", ("failun",) * 4, (_hz("khabn", "tasheeth"),) * 3,
            aruds=(E("A_sahih", "صحيحة", "failun", free=("khabn",)), E("A_tasheeth", "مشعثة", "failun", illa="tasheeth")),
            darbs=(E("D_sahih", "صحيح", "failun", free=("khabn",)), E("D_tasheeth", "مشعث", "failun", illa="tasheeth")),
            pairs={"A_sahih": ("D_sahih", "D_tasheeth"), "A_tasheeth": ("D_sahih", "D_tasheeth")}),
       Form("majzu", "مجزوء", ("failun",) * 3, (_hz("khabn", "tasheeth"),) * 2,
            aruds=(E("A_sahih", "صحيحة", "failun", free=("khabn",)),),
            darbs=(E("D_sahih", "صحيح", "failun", free=("khabn",)), E("D_tadhyil", "مذيل", "failun", illa="tadhyil"),
                   E("D_tarfil", "مرفل", "failun", illa="tarfil")),
            pairs={"A_sahih": ("D_sahih", "D_tadhyil", "D_tarfil")}))

# ---------------------------------------------------------------------------------------------- Shaer labels
NAME_TO_KEY = {m.name: m.key for m in METERS.values()}
SHAER_FORM_TO_KEY = {"تام": "tam", "مجزوء": "majzu", "مخلع": "mukhalla", "أحذ": "ahadh"}


def shaer_target(base_meter: str, form: str) -> tuple[str, str]:
    """(meter key, form key) for a Shaer request such as ("البسيط", "مخلع")."""
    meter = NAME_TO_KEY[base_meter.strip()]
    fkey = SHAER_FORM_TO_KEY.get(form.strip(), "tam")
    if fkey not in METERS[meter].forms:
        raise KeyError(f"{base_meter} has no form {form}")
    return meter, fkey


# ---------------------------------------------------------------------------------------------- enumeration
def hashw_options(form: Form, i: int) -> list[dict]:
    """Realisations of hashw foot i: sound + each allowed zihāf (tashʿīth in المتدارك is a zihāf-like ʿilla)."""
    foot = FEET[form.hashw[i]]
    out = [{"pattern": foot.pattern, "zihaf": None, "rare": False}]
    for z, rare in form.hashw_zihaf[i]:
        try:
            p = apply_illa(foot, "tasheeth") if z == "tasheeth" else apply_zihaf(foot, z)
        except InvalidOperation as exc:
            raise InvalidOperation(f"table error in {form.key} position {i}: {exc}")
        out.append({"pattern": p, "zihaf": z, "rare": rare})
    return out


def _has(z: str | None, part: str) -> bool:
    return bool(z) and z != "tasheeth" and part in ZIHAF_PARTS[z]


def hemistich_analyses(meter_key: str, form_key: str, position: str):
    """Yield every realisation of a sadr ("sadr", with an ʿarūḍ) or ʿajuz ("ajuz", with a ḍarb) of a meter form:
    dicts with pattern, feet (per-foot pattern/zihaf/illa), end form key, rare count and muʿāqaba violations."""
    form = METERS[meter_key].forms[form_key]
    ends = form.aruds if position == "sadr" else form.darbs
    used_ends = set(form.pairs) if position == "sadr" else {d for ds in form.pairs.values() for d in ds}
    hashw = [hashw_options(form, i) for i in range(len(form.hashw) - 1)]
    for end in ends:
        if end.key not in used_ends:
            continue
        for ev in end.variants():
            for combo in itertools.product(*hashw):
                if any(combo[i]["zihaf"] is None for i in form.muraqaba):
                    continue
                feet = [dict(foot=form.hashw[i], **c, illa=None) for i, c in enumerate(combo)]
                feet.append(dict(foot=end.foot, pattern=ev["pattern"], zihaf=ev["zihaf"], illa=ev["illa"], rare=ev["rare"]))
                violations = [(a, b) for a, b in form.muaqaba if _has(feet[a]["zihaf"], "kaff") and _has(feet[b]["zihaf"], "khabn")]
                yield {"meter": meter_key, "form": form_key, "position": position, "end": end.key,
                       "pattern": "".join(f["pattern"] for f in feet), "feet": feet,
                       "rare": sum(bool(f["rare"]) for f in feet), "muaqaba": violations}


def foot_label(f: dict) -> str:
    parts = [FEET[f["foot"]].name]
    if f.get("zihaf"):
        parts.append(ILLA_NAMES["tasheeth"] if f["zihaf"] == "tasheeth" else ZIHAF_NAMES[f["zihaf"]])
    if f.get("illa"):
        parts.append(ILLA_NAMES[f["illa"]])
    return " ".join(parts)
