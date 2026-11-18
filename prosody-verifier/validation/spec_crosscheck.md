# Rule tables: cross-check against independent implementations

Our tables (`prosody/meters.py`) were written first from the classical Khalīlian tables. They were then compared by reading
the source of two open-source implementations: [ARBML Bohour](https://github.com/ARBML/Bohour) (MIT) and
[pyarud](https://github.com/cnemri/pyarud) (MIT). pyarud's tables descend from Bohour's, so the two are not fully
independent. Neither is used at run time. Their code was read, not executed.

The automated checks in `tests/test_feet_meters.py` cover:

* every zihāf and ʿilla gives the named classical foot (52 cases);
* zihāfāt never touch a watad;
* every table entry is a valid operation;
* every hemistich ends on a sākin;
* no two sākin letters meet inside a hemistich;
* the canonical hemistich of each Shaer meter is generated;
* all 21 Shaer (meter, form) labels map to a form.

Two table errors were caught by these checks and fixed: naqṣ/kaff in the majzūʾ wāfir ʿarūḍ/ḍarb, and kaff in the hazaj
ʿarūḍ. Both would end a hemistich on a mutaḥarrik.

## Differences found, and the decision taken

| Meter | Difference | Decision |
|---|---|---|
| الطويل | none in the ʿarūḍ/ḍarb table; both references also allow kharm (ثلم / ثرم) in the first foot | kharm is not modelled yet (rare licence at the start of a verse); listed as a known gap |
| المديد | Bohour pairs the maḥdhūfa ʿarūḍ only with a maqṣūr ḍarb; pyarud with maḥdhūf and abtar | kept the classical table (3 ʿarūḍ, 6 ḍarb: ṣaḥīḥ / maqṣūr, maḥdhūf, abtar / maḥdhūf makhbūn, abtar), a superset of both |
| المديد | shakl in the hashw: both references exclude it | kept, flagged rare |
| البسيط المجزوء | both references pair the maqṭūʿa ʿarūḍ with a ṣaḥīḥ ḍarb | kept the classical pairing maqṭūʿa ↔ maqṭūʿ; corpus calibration will show if human poems disagree |
| البسيط المجزوء | khabn/ṭayy in the ʿarūḍ/ḍarb: not in the references | kept (hashw zihāfāt of مستفعلن entering the end foot) |
| الوافر المجزوء | ʿaql in the ʿarūḍ/ḍarb: not in the references | removed (strict) |
| الكامل | iḍmār on the ḥadhdhāʾ ʿarūḍ: not in the references | kept, flagged rare |
| الكامل | waqṣ/khazl in the ṣaḥīḥ ʿarūḍ/ḍarb: not in the references | kept, flagged rare |
| الهزج | the references allow kaff in the ʿarūḍ | equivalent after the hemistich-final lengthening (مفاعيلُ + ishbāʿ = مفاعيلن), so nothing to add |
| الهزج | qabḍ in the hashw: both references exclude it | kept, flagged rare |
| الرمل المجزوء | khabn on the musabbagh ḍarb: not in the references | kept, flagged rare |
| السريع | both references pair the makhbūla ʿarūḍ with a aṣlam ḍarb too | adopted |
| المنسرح | both references make ṭayy binding in the ʿarūḍ | kept a ṣaḥīḥa ʿarūḍ with free ṭayy/khabn (accepts both) |
| الخفيف | Bohour has no khabn on the ṣaḥīḥ ḍarb; pyarud has it | kept khabn (pyarud agrees; common in practice) |
| الخفيف | the maḥdhūfa ʿarūḍ (↔ maḥdhūf ḍarb) is merged into other entries in the references | kept as its own entry (classical) |
| المجتث | tashʿīth in the ʿarūḍ: not in the references | kept; corpus calibration decides |
| المتقارب | both references pair the maḥdhūfa ʿarūḍ with every ḍarb (ṣaḥīḥ, maqṣūr, ...) | kept the classical pairing (maḥdhūfa ↔ maḥdhūf, abtar); corpus calibration will show if this is too strict |
| المضارع، المقتضب | murāqaba: the first foot may not stay sound | added (`muraqaba`) |
| all | taṣrīʿ: in the opening bayt the ʿarūḍ may take the form of the ḍarb | applied by the scanner for the first bayt |

The rows marked "corpus calibration" are the open questions. They are settled by the human poems in the Shaer data (see
`validation/` reports), and then confirmed by a prosody expert.
