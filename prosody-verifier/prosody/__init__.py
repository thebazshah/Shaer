"""Formal Arabic prosody (ʿarūḍ and qāfiya) for verifying generated poems.

Modules:
  feet       the eight base feet (tafāʿīl) and the operations that derive their variants (zihāfāt and ʿilal)
  meters     the sixteen meters, their forms (tām, majzūʾ, mashṭūr, manhūk, mukhallaʿ, aḥadhdh) and arūḍ/ḍarb tables
  arudi      prosodic writing: text (fully, partly or not diacritised) -> lattice of mutaḥarrik/sākin choices
  scan       matching a hemistich or bayt lattice against the meter patterns, with diagnostics
  qafiya     rhyme analysis (rawī, waṣl, khurūj, ridf, taʾsīs, dakhīl) and refrain (Urdu-style radeef) detection
  verify     poem-level verification combining all of the above

Notation: a pattern is a string over {"1", "0"}: 1 = mutaḥarrik (a letter carrying a short vowel), 0 = sākin (a letter
without a vowel, including the letters of long vowels). This is the classical /, o notation of ʿarūḍ.
"""
