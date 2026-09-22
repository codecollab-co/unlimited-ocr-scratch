# Diligo — Founding Plan

Locked via a structured decision interview: idea → phases → feasibility → tech stack →
implementation → branding → name/mark. This is the single source of truth for what was
decided and why — every "chosen over X" below records a real alternative that was
considered and the reason it lost. Preserve that reasoning if this plan is revised;
don't silently re-litigate a decision without knowing why it was made the first time.

## Idea

- **Product**: AI legal-health / diligence-readiness tool for pre-seed/seed founders.
  Ingests formation documents (cap table, SAFEs, option pool, IP assignments) and
  produces a continuously-updated legal health report.
- **Trust model**: AI + licensed attorney sign-off on every formal deliverable — not
  pure AI output. Defuses UPL (unauthorized practice of law) risk and matches a16z's
  2026 thesis: AI does the work, a human signs off, price ~80% below pure-human legal
  services.
- **Distribution**: accelerator channel first — 100X.VC, Sequoia Surge (India), Hub71
  (Dubai) — not direct-to-founder or law-firm white-label. One accelerator relationship
  delivers dozens of founders at once instead of one-by-one acquisition.
- **Document scope**: narrow at launch (formation docs only), broadening to commercial
  contracts/IP next, then full legal surface area.
- **Pricing**: hybrid — low-cost subscription for continuous monitoring, plus a larger
  one-time fee for an attorney-signed "diligence-ready" report at raise time.

## Phases

- **Phase 1 wedge**: cap table reconciliation — pure arithmetic cross-referencing, the
  most mechanically verifiable, least judgment-dependent check available.
- **Build order**: cap table reconciliation → IP assignment completeness → everything
  else.
- **Pilot structure**: 3 parallel design partners (100X.VC, Sequoia Surge, Hub71), not
  one sequential pilot.
- **Phase 1 exit criterion**: outcome-based — a minimum number of real,
  attorney-confirmed cap-table errors caught. Not usage volume, not revenue.
- **Timeline**: aggressive, 6-8 weeks — ship fast, close gaps with AI iteration rather
  than waiting for full coverage.

## Feasibility

- **Attorney capacity**: contract/freelance attorneys per jurisdiction (Indian company
  law; DIFC/ADGM UAE law) — not an in-house hire, not a law-firm partnership. Scales
  cost with real pilot volume instead of committing fixed cost up front.
- **OCR/ML foundation**: builds on the from-scratch Rust `unlimited-ocr-scratch`
  reimplementation, not a commercial OCR/document-AI API. Real tradeoff accepted: that
  project's end-to-end generation path (multimodal input splicing, tokenizer/chat
  template handling) was explicitly left unfinished as a research build, and now has to
  be completed inside the 6-8 week window.
- **Pipeline scope**: full generative decode pipeline, not a minimum extraction-only
  slice — a second scope-adding call on the same timeline.
- **Inference hosting**: Hugging Face Inference Endpoints — natural fit since candle,
  the Rust ML framework already in use, is HF's own.
- **ML engineering**: in-house, one dedicated team member — no external ML contractor.
- **Funding**: bootstrapped. Phase 1 costs (contract attorneys, pay-per-use inference)
  are small enough not to justify raising, or applying to an accelerator purely for
  stipend funding.
- **Data handling**: no raw document persistence. Cap table/SAFE documents are
  processed ephemerally; only extracted structured data and the final report are
  stored. Sidesteps most of India's DPDP Act and UAE data-residency exposure, and is a
  strong trust signal for a founder handing over their most sensitive pre-raise
  documents.

## Tech stack

- **Frontend**: Next.js, deployed on Vercel.
- **Backend**: Python/FastAPI, deployed on Railway. Owns all computation (reconciliation
  logic, characterization) and is the only service that talks to the Rust inference
  service. Chosen over an all-Rust backend (too slow to ship SaaS CRUD/billing/auth
  under an aggressive deadline in a thin ecosystem) and over Convex-only (would move
  the heavy-lifting logic into TypeScript, against the explicit preference to keep
  computation backend-owned).
- **ML inference**: Rust/candle (`unlimited-ocr-scratch`), on Hugging Face Inference
  Endpoints.
- **Live state / reactive data**: Convex — owns the attorney-review queue and dashboard
  status, not the core reconciliation computation.
- **Payments**: Dodo Payments, not Stripe. Confirmed official TypeScript *and* Python
  SDKs, a `@dodopayments/nextjs` package, and an official FastAPI boilerplate. Chosen
  specifically because Dodo is a merchant-of-record built for the India↔Dubai
  cross-border case this company actually has.
- **Database**: superseded by Convex for live/reactive state. FastAPI's computation
  layer does not currently own a separate SQL database — **open**, revisit if/when
  Phase 2+ needs persistent relational storage beyond what Convex models well.
- **Auth**: roll-your-own — NextAuth/Auth.js plus custom JWT verification in FastAPI and
  Convex. Flagged as real, non-trivial setup work across three services; chosen over
  the lower-risk managed option (Clerk) as an explicit call.

## Implementation

- **Build sequencing**: parallel, not sequential. The ML engineer finishes the Rust
  generative pipeline while the rest of the team builds the product shell
  (Next.js/FastAPI/Convex/auth) simultaneously, against an API contract agreed in week 1.
- **Attorney review workflow**: full in-app dashboard (Convex-backed live queue), not a
  manual email-based process — a fourth real workstream stacked onto the 6-8 week build
  alongside the generative pipeline and the three-service auth.
- **Rollout**: staggered by jurisdiction. The two India pilots (100X.VC + Sequoia Surge)
  launch together first, sharing Indian company law and one contracted attorney; Hub71
  comes online once the India cohort has proven the product and the separate UAE/DIFC
  attorney relationship is in place.

## Branding

- **Tone**: hybrid — credible/trustworthy enough to be handed sensitive legal
  documents, but warm and direct rather than stiff. No jargon for its own sake; trust
  signals (attorney sign-off, credentials) presented alongside a founder-to-founder
  voice.
- **Naming metaphor**: clarity/insight — chosen over protection/shield (too common in
  security/compliance branding). Names the mechanism (seeing through complexity at
  unbounded scale) as much as the emotional payoff.

## Name & mark system

Full brand system — name rationale, palette, type, the three locked marks, and the
animated loading state — lives in `../_decision_docs/brand-system.md`. Production
files are in `../assets/`.

## Cumulative risk register (as of this plan)

Four real workstreams are stacked into the same 6-8 week window, each individually
justified but compounding in total: (1) finishing the unbuilt Rust generative pipeline,
(2) three-service roll-your-own auth, (3) a full in-app attorney-review dashboard, (4)
the parallel-build coordination overhead itself. None were reversed on review — each was
a deliberate call — but if the timeline slips, this is where to look first.
