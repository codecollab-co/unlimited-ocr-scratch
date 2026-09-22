# Diligo — Brand System (Locked)

## Name

**Diligo** — Latin root of "diligence." Names the product's actual outcome
(diligence-*readiness*) rather than a supporting metaphor.

Checked clear of trademark/domain/company-name collisions. Ruled out along the way,
in order, with reason:

| Name | Why it was dropped |
|---|---|
| FounderShield *(working name only)* | "Shield/protect" is one of the most overused metaphors in security/compliance branding (Vanta, Secureframe, Lemonade, ...) |
| Lucidus | `lucidus.com` owned by an active IT/cybersecurity firm |
| Lucidate | Taken by two active AI companies (Lucidate Ltd, UK; Lucidate AI, code intelligence) |
| Lucide | Collides with `lucide-react`, the icon library this exact Next.js stack is likely to import |
| Clarion | Taken by Clarion AI Partners — an existing AI + Law firm, direct category collision |
| Certus | Taken by a YC-backed AI trademark-law agent, among several others |
| Diligend, Verascope, Clarivue, Truvantage | Each had a live or phonetically-close collision |

**Diligo** returned zero collisions across trademark, domain, and company-name searches.

## Palette

| Token | Light | Dark |
|---|---|---|
| Ink | `#16233D` | `#EDF2F1` |
| Accent (signal) | `#0E9C97` | `#2FE0D2` |
| Paper | `#F1F5F4` | `#0D1424` |

## Type

- **Display / editorial**: `ui-serif, 'Iowan Old Style', 'Palatino Linotype', Georgia, serif`
- **Body / UI**: `ui-sans-serif, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif`
- **Marks (wordmark/lettermark)**: `'Century Gothic', 'Avenir Next', 'Futura', ui-sans-serif, sans-serif`

## The three locked marks

| Mark | Variant | Usage |
|---|---|---|
| Wordmark | C2 — "Diligo," bold, final *o* as an open ring | Text-only contexts: legal footer, plain-text signature, anywhere a symbol can't render |
| Combination mark | C2 — radar-ring icon + wordmark | **Primary mark — website, product nav, marketing, decks** |
| Lettermark | C1 — solid accent tile, reversed D | **Favicon, app icon, avatar** — anywhere the combination mark is too fine-detailed to survive scaling down |

## Loading state

The combination mark's radar-ring icon doubles as the product's loading indicator in
place of a generic shimmer or spinner: the three rings pulse outward from the center
dot, staggered ~0.45s apart, looping. Respects `prefers-reduced-motion` (falls back to
a static, faded ring state — no motion). See `../assets/icon-radar-loading.svg` (icon
alone) and `../assets/combination-mark-loading.svg` (full lockup, loading variant).

## Files (`../assets/`)

- `lettermark-favicon-app-icon.svg` — favicon / app icon / avatar
- `combination-mark-primary.svg` — primary website lockup, static
- `combination-mark-loading.svg` — primary lockup, animated loading state
- `wordmark.svg` — standalone text-context mark
- `icon-radar-static.svg` — icon alone, static
- `icon-radar-loading.svg` — icon alone, animated

Full exploration history (9 initial directions → 3 refinements per family → this locked
system) is preserved at: https://claude.ai/code/artifact/ecb91c59-d05d-4026-9cc2-844cc0ff38c5

## Production note

The wordmark and lettermark SVGs render "D"/"Diligo" via `<text>` elements on a system
font stack, for portability without embedding font files. Before production or print
use, outline the type in a vector tool (Figma/Illustrator) so rendering doesn't depend
on which fonts happen to be installed on the viewing system. Platform app-icon exports
(iOS/Android require specific raster sizes and safe-zone insets) still need generating
from these SVGs when that becomes relevant — not done here.
