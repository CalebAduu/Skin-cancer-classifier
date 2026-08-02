import PredictForm from "./predict-form";

export default function Home() {
  return (
    <div className="flex flex-1 flex-col items-center">
      <main className="flex w-full max-w-2xl flex-1 flex-col gap-9 px-6 py-16 sm:py-24">
        <header className="flex flex-col gap-5">
          <div className="flex items-center gap-3.5">
            <span
              className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl text-white shadow-lg"
              style={{
                background: "linear-gradient(135deg, var(--accent), var(--accent-2))",
                boxShadow: "0 8px 20px -6px color-mix(in srgb, var(--accent) 50%, transparent)",
              }}
            >
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="9" />
                <path d="M9 12a3 3 0 1 1 6 0c0 1.5-1.5 2-1.5 3.5" />
                <circle cx="13.5" cy="18" r="0.5" fill="currentColor" />
              </svg>
            </span>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight text-[var(--text-primary)]">
                Dermoscopy Classifier
              </h1>
              <p className="text-xs font-medium tracking-wide text-[var(--text-muted)]">
                Hybrid CNN&ndash;ViT &middot; HAM10000 &middot; Score-CAM
              </p>
            </div>
          </div>

          <p className="text-sm leading-6 text-[var(--text-secondary)]">
            Fuses a dermoscopy image with patient metadata (age, sex, site)
            through cross-attention adapters to produce calibrated,
            temperature-scaled probabilities across seven lesion types.
          </p>

          <div className="flex items-start gap-2.5 rounded-xl border border-[var(--status-warning)]/30 bg-[var(--status-warning)]/10 px-3.5 py-3 text-xs leading-5 text-[var(--text-primary)]">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--status-warning)"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="mt-0.5 shrink-0"
            >
              <path d="M12 9v4M12 17h.01" />
              <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
            </svg>
            <span>
              <strong className="font-semibold">Research prototype, not a diagnostic device.</strong>{" "}
              Trained on HAM10000; performance degrades under real-world domain
              shift. Not evaluated for clinical use.
            </span>
          </div>
        </header>

        <PredictForm />
      </main>

      <footer className="w-full max-w-2xl px-6 pb-10 text-center text-xs text-[var(--text-muted)]">
        Decision-support research prototype &middot; not for clinical use
      </footer>
    </div>
  );
}
