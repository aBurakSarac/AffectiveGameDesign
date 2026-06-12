# Translation Guide — La Façade Fissurée study site

This site ships an i18n runtime and three language dictionaries. The English
dictionary (`lang/en.js`) is **the canonical source of truth** (501 keys).

> **Extraction is complete.** Every user-facing string in the site is already
> keyed and wired to `t(...)`. Your only job is to **translate the values** in
> `lang/fr.js` (French) and `lang/tr.js` (Turkish). Do not touch the `.jsx`
> source, `site-data.js`, or `lang/en.js`.

`fr.js` and `tr.js` currently hold the English text as placeholders (each value
marked by being identical to `en.js`). Replace each value with its translation,
following the rules below. Keep the keys exactly as they are.

---

## 1. How the system works

- `i18n.js` defines `window.I18N` with:
  - `I18N.t(key, fallback)` → resolves a string for the current language,
    falling back to English, then to `fallback`/`key`.
  - `I18N.tSlot(str, slot, node)` → splits a string on a `{slot}` placeholder
    and returns `[before, node, after]` for JSX (see §3).
  - `I18N.set(code)` / `I18N.lang` / `I18N.onChange(fn)` — language state,
    persisted in `localStorage` under `facadeLang_v1`.
- `lang/en.js`, `lang/fr.js`, `lang/tr.js` each call
  `window.I18N._register("<code>", { ...flat dotted keys... })`.
- Load order (already wired in `index.html`): `i18n.js` → `lang/en.js` →
  `lang/fr.js` → `lang/tr.js` → `site-data.js` → components → `dist/site-app.js`.
- Components read text with `window.I18N.t("key")`. React re-renders on language
  change because `SiteApp` keeps `lang` in state (`site-app.jsx`).

**Do not edit `i18n.js`.** Only edit the `lang/*.js` files (and, while
extracting, the component/`site-data.js` source — see §5).

---

## 2. Key naming

Flat, dotted, grouped by area: `"<area>.<thing>"`.
Examples already present: `nav.overview`, `hero.title`, `hero.sub`.
When you extract a new string, **add the key to all three** `lang/*.js` files
(English value in `en.js`; English placeholder + `TODO` in `fr.js`/`tr.js`
until translated).

---

## 3. `{slot}` placeholders — the important rule

Some sentences contain an **emphasized / styled word** that is rendered as
separate JSX (italic, coloured, a `<b className="mono">`, etc.). To translate
the sentence without breaking the markup, the styled word is pulled out:

```
"hero.title":   "Can a plain webcam feel a player's {fear}?"
"hero.fearWord": "fear"
```

The component renders it with:

```jsx
I18N.tSlot(t("hero.title"), "fear", <em>{t("hero.fearWord")}</em>)
```

**Rules for `{slot}` tokens:**
- Keep the token **verbatim** (`{fear}`) — never translate or delete it.
- You **may move** the token to a different position in the sentence to fit the
  target language's grammar. (Turkish, for instance, often reorders.)
- Translate the **pulled-out word** (`hero.fearWord`) separately, preserving the
  sentence's intended meaning. The styled word and the sentence around it form
  one thought — keep that context intact.

Slots already in the dictionaries: `{fear}` (hero.title), `{f12}`/`{f15}`
(fusion.lead), `{file}` (rail.note), `{shown}`/`{total}` (cust.subtitle). Some
values instead carry inline HTML (`<b>`, `<em>`, `<i>`) — e.g. `m3.p`, `m4.p`,
`m4.key`, `bg.lit.*.d`, `report.res.asymCap`, the `report.game.step*`,
`bg.swotNote`, `outlook.conclTitle`, `data.path.2.d`. **Keep those tags
verbatim**; translate only the words between them.

### ⚠ markers — read these before translating a group

`lang/*.js` has `⚠` comments directly above the keys that are risky to
translate in isolation. Two kinds:

- **SPLIT sentence** — one thought spread across several keys for styling (e.g.
  `hero.title` + `hero.fearWord`; `fusion.lead` + `fusion.leadF12` +
  `fusion.leadF15`; `data.fuse.*.eq0` + `eq1`). The comment prints the full
  assembled sentence. Translate the whole group **together** so it reads as one
  coherent sentence, keeping the `{slot}`/tag tokens.
- **ASSEMBLED** — a key joined at runtime to a number or value, where word order
  matters (e.g. `methods.label` + `" 01"`; `fusion.case` + `" 1"`;
  `hud.amp.rest`/`now` + a BPM number; `rail.meta` preceded by the clip count;
  `hud.primary.distPrefix` followed by the dominant emotion). The comment shows
  the assembled result; phrase your translation so the inserted value lands
  naturally.

There are 13 such markers. Never translate one of these keys without reading its
`⚠` comment first.

---

## 4. Do NOT translate these tokens

Proper nouns, model names, metrics, and code-ish tokens stay as-is in every
language:

`La Façade Fissurée`, `FER`, `rPPG`, `HSEmotion`, `EfficientNet-B0`,
`MediaPipe`, `FaceLandmarker`, `POS`, `CONSENSUS`, `CLAHE`, `HUD`, `F12`, `F15`,
`BPM`, `ROI`, `Haar`, `NavMesh`, `Unity`, `Galatasaray University`, author/advisor
names, section ids, units (`ms`, `fps`, `%`, `s`). Keep glyphs `▸`, `·`, `×`, `→`.

Numbers and figures (`0.77`, `99.9%`, `53 ms`) are not translated, but
surrounding words are.

---

## 5. Key groups (all already extracted)

Every group below is fully keyed in `lang/en.js` and wired in the source. You
only translate the values in `fr.js` / `tr.js`.

| Prefix | Where it shows |
| --- | --- |
| `nav.*`, `section.*` | top bar, side rail, customizer labels |
| `hero.*` | overview / hero |
| `data.*` | content from `site-data.js` (results, pipeline, RQs, findings, enemies, design path…) |
| `hud.*`, `emo.*` | the live HUD (telemetry, verdict, signals, amplifiers, formula chain) |
| `player.*`, `rail.*`, `light.*` | HUD player section + session rail |
| `fusion.*`, `teach.*` | F12-vs-F15 section |
| `methods.*`, `m1.*`–`m4.*` | the four method animations |
| `report.*` | pipeline / results / game / design-path sections |
| `bg.*`, `perf.*`, `outlook.*`, `gloss.*` | background, performance, outlook, glossary |
| `footer.*`, `cust.*` | footer + section customizer |

**Workflow:** edit only `lang/fr.js` and `lang/tr.js`. No build step — they are
plain scripts. After translating, just reload the page and switch language with
the EN/FR/TR control in the top bar. Untranslated keys fall back to English, so
you can work incrementally.

**Sanity check** (keys stay identical across the three files):

```bash
cd Website
for f in lang/en.js lang/fr.js lang/tr.js; do
  printf "%s: " "$f"; grep -coE '^\s*"[^"]+":' "$f"
done   # all three must print the same number (501)
```

Do **not** edit `.jsx`, `dist/*.js`, `site-data.js`, or `i18n.js`. If a new
English string ever needs adding, it goes in `en.js` first, then `fr.js`/`tr.js`.

---

## 6. Tone

- **French**: formal register (vous-form where a verb addresses the reader),
  academic but readable — this is a graduation-defense site.
- **Turkish**: formal/academic register; keep sentences tight.
- Keep the slightly punchy, plain-language voice of the English (e.g. "A fearful
  face can lie."). Don't over-formalize the marketing lines.
