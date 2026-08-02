"use client";

import { Fragment, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

const LOCALIZATIONS = [
  "back", "trunk", "lower extremity", "upper extremity", "chest", "abdomen",
  "face", "scalp", "neck", "hand", "foot", "acral", "ear", "genital", "unknown",
];

// History & symptom questions, styled after the Glasgow 7-point checklist
// (3 major signs worth 2 points, 4 minor signs worth 1 point; total >= 3
// suggests specialist review). Deliberately steers clear of anything the CNN
// already reads from the pixels (asymmetry, border, colour, size) — those
// are redundant to ask a layperson to self-assess and the model already
// extracts them. These questions instead capture what a single static photo
// cannot: time course and change over time, symptoms, and personal risk.
// Purely a UI checklist — never sent to the model or blended into its
// probabilities; the trained model's embedding tables are sized to exactly
// the 12 features it learned on, so there is no slot for this signal even if
// we wanted to feed it in. Shown as a separate, transparently rule-based
// flag, motivated by the model's comparatively weak melanoma sensitivity
// (~71% on held-out test data, the lowest of the three malignant classes).
const DURATION_OPTIONS = ["<1 month", "1–6 months", "6–12 months", ">1 year", "not sure"] as const;
const YES_NO_UNSURE = ["yes", "no", "not sure"] as const;

const HISTORY_QUESTIONS = [
  {
    id: "duration",
    weight: "context" as const,
    question: "How long have you noticed it?",
    options: DURATION_OPTIONS,
  },
  {
    id: "onset",
    weight: "context" as const,
    question: "Did it start as a mole, a wound/injury, or appear out of nowhere?",
    options: ["a mole", "a wound or injury", "appeared out of nowhere", "not sure"] as const,
  },
  {
    id: "evolving",
    weight: "major" as const,
    question: "Has it changed in the last few months?",
    options: YES_NO_UNSURE,
  },
  {
    id: "bleeding",
    weight: "minor" as const,
    question: "Has it bled, oozed, or crusted?",
    options: YES_NO_UNSURE,
  },
  {
    id: "nonhealing",
    weight: "minor" as const,
    question: "Is there a sore or spot that won't heal?",
    options: YES_NO_UNSURE,
  },
  {
    id: "sensation",
    weight: "minor" as const,
    question: "Is it itchy, tender, or painful?",
    options: YES_NO_UNSURE,
  },
  {
    id: "uglyDuckling",
    weight: "minor" as const,
    question: "Does it look different from your other moles?",
    options: ["yes", "no", "I don't have others", "not sure"] as const,
  },
  {
    id: "riskHistory",
    weight: "context" as const,
    question: "Any personal or family history of skin cancer?",
    options: YES_NO_UNSURE,
  },
];

const CHANGE_OPTIONS = [
  { value: "bigger", label: "Got bigger" },
  { value: "colour", label: "Changed colour" },
  { value: "shape", label: "Changed shape" },
  { value: "raised", label: "Became raised" },
];

function computeHistoryFlag(answers: Record<string, string>, changeDetails: string[]) {
  const majorHits: string[] = [];
  const minorHits: string[] = [];

  if (answers.evolving === "yes") majorHits.push("changed recently");
  if (changeDetails.length > 0) {
    majorHits.push(
      changeDetails
        .map((v) => CHANGE_OPTIONS.find((o) => o.value === v)?.label.toLowerCase())
        .filter(Boolean)
        .join(", ")
    );
  }
  if (answers.bleeding === "yes") minorHits.push("bled, oozed, or crusted");
  if (answers.nonhealing === "yes") minorHits.push("won't heal");
  if (answers.sensation === "yes") minorHits.push("itchy, tender, or painful");
  if (answers.uglyDuckling === "yes") minorHits.push("looks different from other moles");

  const contextHits: string[] = [];
  if (answers.duration) contextHits.push(`noticed ${answers.duration}`);
  if (answers.onset) contextHits.push(`started as ${answers.onset}`);
  if (answers.riskHistory === "yes") contextHits.push("personal/family history of skin cancer");

  const score = majorHits.length * 2 + minorHits.length * 1;
  return { majorHits, minorHits, contextHits, score, flagged: score >= 3 };
}

const CLASS_INFO: Record<string, { name: string; malignant: boolean }> = {
  NV: { name: "Melanocytic nevus", malignant: false },
  MEL: { name: "Melanoma", malignant: true },
  BCC: { name: "Basal cell carcinoma", malignant: true },
  AKIEC: { name: "Actinic keratosis / intraepithelial carcinoma", malignant: true },
  BKL: { name: "Benign keratosis", malignant: false },
  DF: { name: "Dermatofibroma", malignant: false },
  VASC: { name: "Vascular lesion", malignant: false },
};

interface PredictResult {
  argmax_label: string;
  calibrated_label?: string;
  probabilities: Record<string, number>;
  malignant_probability: number;
  temperature: number;
  latency_ms: number;
  disclaimer: string;
  scorecam_png_base64?: string;
  ood_distance?: number;
  ood_threshold?: number;
  likely_out_of_distribution?: boolean;
}

function meterStatus(p: number): "good" | "warning" | "critical" {
  if (p >= 0.4) return "critical";
  if (p >= 0.1) return "warning";
  return "good";
}

const STATUS_COLOR: Record<string, string> = {
  good: "var(--status-good)",
  warning: "var(--status-warning)",
  critical: "var(--status-critical)",
};

function ConfidenceRing({ value, color }: { value: number; color: string }) {
  const size = 56;
  const stroke = 5;
  const r = (size - stroke) / 2;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - value);
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke="var(--gridline)"
        strokeWidth={stroke}
      />
      <circle
        className="ring-progress"
        style={{ ["--ring-circumference" as string]: circumference }}
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
      />
    </svg>
  );
}

function Spinner() {
  return (
    <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" opacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function RadioPills({
  name,
  options,
  value,
  onChange,
}: {
  name: string;
  options: readonly string[];
  value: string | undefined;
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((opt) => (
        <label
          key={opt}
          className={`cursor-pointer rounded-full border px-2.5 py-1 text-xs transition-colors ${
            value === opt
              ? "border-[var(--accent)] bg-[var(--accent)] text-white"
              : "border-[var(--gridline)] text-[var(--text-secondary)] hover:border-[var(--accent)]"
          }`}
        >
          <input
            type="radio"
            name={name}
            value={opt}
            checked={value === opt}
            onChange={() => onChange(opt)}
            className="sr-only"
          />
          {opt}
        </label>
      ))}
    </div>
  );
}

export default function PredictForm() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("unknown");
  const [localization, setLocalization] = useState("unknown");
  const [explain, setExplain] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [changeDetails, setChangeDetails] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function onFileChange(f: File | null) {
    setFile(f);
    setResult(null);
    setError(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
  }

  function setAnswer(id: string, value: string) {
    setAnswers((prev) => ({ ...prev, [id]: value }));
  }

  function toggleChangeDetail(value: string) {
    setChangeDetails((prev) =>
      prev.includes(value) ? prev.filter((v) => v !== value) : [...prev, value]
    );
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f) onFileChange(f);
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) {
      setError("Choose an image first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", file);
    if (age) formData.append("age", age);
    formData.append("sex", sex);
    formData.append("localization", localization);
    formData.append("explain", String(explain));

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text}`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  const sortedProbs = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];
  const topLabel = result?.calibrated_label ?? result?.argmax_label ?? "";
  const topInfo = CLASS_INFO[topLabel];
  const malignantStatus = result ? meterStatus(result.malignant_probability) : "good";
  const history = computeHistoryFlag(answers, changeDetails);

  return (
    <div className="flex flex-col gap-6">
      <form
        onSubmit={onSubmit}
        className="flex flex-col gap-5 rounded-3xl border border-[var(--border-hairline)] bg-[var(--surface-1)] p-6 shadow-[var(--shadow-card)]"
      >
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`group flex cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-4 py-9 text-center transition-all ${
            dragActive
              ? "scale-[1.01] border-[var(--accent)] bg-[var(--accent-track)]/30"
              : "border-[var(--gridline)] hover:border-[var(--accent)] hover:bg-[var(--accent-track)]/10"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
            className="hidden"
          />
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={previewUrl}
              alt="Selected dermoscopy image"
              className="max-h-56 rounded-xl border border-[var(--border-hairline)] object-contain shadow-sm"
            />
          ) : (
            <span
              className="flex h-12 w-12 items-center justify-center rounded-full transition-transform group-hover:scale-105"
              style={{ background: "var(--accent-track)" }}
            >
              <svg
                width="22"
                height="22"
                viewBox="0 0 24 24"
                fill="none"
                stroke="var(--accent)"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <circle cx="9" cy="9" r="2" />
                <path d="m21 15-5-5L5 21" />
              </svg>
            </span>
          )}
          <div className="text-sm text-[var(--text-secondary)]">
            {file ? (
              <span className="font-medium text-[var(--text-primary)]">{file.name}</span>
            ) : (
              <>
                <span className="font-medium text-[var(--accent)]">Click to upload</span> or
                drag a dermoscopy image here
              </>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-[var(--text-secondary)]">Age</label>
            <input
              type="number"
              min={0}
              max={100}
              value={age}
              onChange={(e) => setAge(e.target.value)}
              placeholder="optional"
              className="rounded-xl border border-[var(--gridline)] bg-[var(--surface-2)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-[var(--text-secondary)]">Sex</label>
            <select
              value={sex}
              onChange={(e) => setSex(e.target.value)}
              className="rounded-xl border border-[var(--gridline)] bg-[var(--surface-2)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
            >
              <option value="unknown">unknown</option>
              <option value="male">male</option>
              <option value="female">female</option>
            </select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs font-medium text-[var(--text-secondary)]">
              Localization
            </label>
            <select
              value={localization}
              onChange={(e) => setLocalization(e.target.value)}
              className="rounded-xl border border-[var(--gridline)] bg-[var(--surface-2)] px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
            >
              {LOCALIZATIONS.map((loc) => (
                <option key={loc} value={loc}>
                  {loc}
                </option>
              ))}
            </select>
          </div>
        </div>

        <label className="flex cursor-pointer items-center justify-between rounded-xl border border-[var(--gridline)] px-3 py-2.5">
          <span className="flex flex-col">
            <span className="text-sm text-[var(--text-primary)]">Score-CAM explanation</span>
            <span className="text-xs text-[var(--text-muted)]">
              CPU inference: can take several minutes
            </span>
          </span>
          <span className="relative inline-flex h-6 w-11 shrink-0 items-center">
            <input
              type="checkbox"
              checked={explain}
              onChange={(e) => setExplain(e.target.checked)}
              className="peer sr-only"
            />
            <span className="absolute inset-0 rounded-full bg-[var(--gridline)] transition-colors peer-checked:bg-[var(--accent)]" />
            <span className="absolute left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform peer-checked:translate-x-5" />
          </span>
        </label>

        <div className="flex flex-col gap-3 rounded-xl border border-[var(--gridline)] p-3">
          <button
            type="button"
            onClick={() => setShowHistory((v) => !v)}
            className="flex w-full cursor-pointer items-center justify-between text-left"
          >
            <span className="flex flex-col">
              <span className="text-sm text-[var(--text-primary)]">
                History &amp; symptoms (optional)
              </span>
              <span className="text-xs text-[var(--text-muted)]">
                Catches what a single photo can&rsquo;t.
              </span>
            </span>
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="var(--text-muted)"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`shrink-0 transition-transform ${showHistory ? "rotate-180" : ""}`}
            >
              <path d="m6 9 6 6 6-6" />
            </svg>
          </button>

          {showHistory && (
            <div className="flex flex-col gap-4 border-t border-[var(--gridline)] pt-3">
              {HISTORY_QUESTIONS.map((q) => (
                <Fragment key={q.id}>
                  <div className="flex flex-col gap-1.5">
                    <span className="text-xs text-[var(--text-primary)]">{q.question}</span>
                    <RadioPills
                      name={q.id}
                      options={q.options}
                      value={answers[q.id]}
                      onChange={(v) => setAnswer(q.id, v)}
                    />
                  </div>
                  {q.id === "evolving" && answers.evolving === "yes" && (
                    <div className="flex flex-col gap-1.5 border-l-2 border-[var(--gridline)] pl-3">
                      <span className="text-xs text-[var(--text-primary)]">
                        How did it change? (pick any)
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {CHANGE_OPTIONS.map((opt) => (
                          <label
                            key={opt.value}
                            className={`cursor-pointer rounded-full border px-2.5 py-1 text-xs transition-colors ${
                              changeDetails.includes(opt.value)
                                ? "border-[var(--accent)] bg-[var(--accent)] text-white"
                                : "border-[var(--gridline)] text-[var(--text-secondary)] hover:border-[var(--accent)]"
                            }`}
                          >
                            <input
                              type="checkbox"
                              checked={changeDetails.includes(opt.value)}
                              onChange={() => toggleChangeDetail(opt.value)}
                              className="sr-only"
                            />
                            {opt.label}
                          </label>
                        ))}
                      </div>
                    </div>
                  )}
                </Fragment>
              ))}
            </div>
          )}
        </div>

        <button
          type="submit"
          disabled={loading || !file}
          className="flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold text-white shadow-md transition-all enabled:hover:-translate-y-0.5 enabled:hover:shadow-lg disabled:opacity-40"
          style={{ background: "linear-gradient(135deg, var(--accent), var(--accent-2))" }}
        >
          {loading && <Spinner />}
          {loading ? "Running inference..." : "Predict"}
        </button>

        {error && (
          <div className="flex items-start gap-2 rounded-xl border border-[var(--status-critical)]/30 bg-[var(--status-critical-track)] px-3 py-2 text-sm text-[var(--status-critical)]">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="mt-0.5 shrink-0"
            >
              <circle cx="12" cy="12" r="9" />
              <path d="M12 8v5M12 16h.01" />
            </svg>
            {error}
          </div>
        )}
      </form>

      {result && (
        <div className="animate-fade-in-up flex flex-col gap-5 rounded-3xl border border-[var(--border-hairline)] bg-[var(--surface-1)] p-6 shadow-[var(--shadow-card)]">
          {result.likely_out_of_distribution && (
            <div
              className="flex items-start gap-3 rounded-2xl p-4 text-white shadow-md"
              style={{ background: "linear-gradient(135deg, var(--status-critical), #8a2424)" }}
            >
              <svg
                width="22"
                height="22"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="mt-0.5 shrink-0"
              >
                <path d="M21 21 3 3" />
                <path d="M10.5 5H19a2 2 0 0 1 2 2v8.5" />
                <path d="M5 8v8a2 2 0 0 0 2 2h8" />
                <circle cx="9" cy="10" r="1.5" />
                <path d="m21 15-4.5-4.5a2 2 0 0 0-2.8 0L5 19" />
              </svg>
              <div className="flex flex-col gap-1">
                <p className="text-sm font-semibold">
                  This doesn&rsquo;t look like a dermoscopy image
                </p>
                <p className="text-xs leading-5 text-white/90">
                  The image is a poor match for real dermoscopy images the
                  model was trained on, so the classification below is
                  unreliable and shouldn&rsquo;t be trusted. Try a close-up,
                  well-lit photo of a single skin lesion.
                </p>
              </div>
            </div>
          )}

          <div
            className={`flex flex-col gap-5 ${result.likely_out_of_distribution ? "opacity-50" : ""}`}
          >
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-center gap-4">
              <div className="relative shrink-0">
                <ConfidenceRing
                  value={result.probabilities[topLabel] ?? 0}
                  color={topInfo?.malignant ? "var(--status-critical)" : "var(--accent)"}
                />
                <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-[var(--text-primary)]">
                  {Math.round((result.probabilities[topLabel] ?? 0) * 100)}%
                </span>
              </div>
              <div>
                <p className="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
                  Calibrated prediction
                </p>
                <p className="text-2xl font-semibold text-[var(--text-primary)]">
                  {topInfo?.name ?? topLabel}
                </p>
              </div>
            </div>
            <span className="shrink-0 rounded-full bg-[var(--surface-2)] px-2.5 py-1 text-xs font-mono text-[var(--text-muted)]">
              {result.latency_ms.toLocaleString()} ms
            </span>
          </div>

          <div
            className="flex flex-col gap-1.5 rounded-xl p-3.5"
            style={{
              background: `color-mix(in srgb, ${STATUS_COLOR[malignantStatus]} 6%, var(--surface-2))`,
            }}
          >
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-1.5 font-medium text-[var(--text-primary)]">
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke={STATUS_COLOR[malignantStatus]}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  {malignantStatus === "good" ? (
                    <path d="m5 12 5 5 9-9" />
                  ) : (
                    <>
                      <path d="M12 9v4M12 17h.01" />
                      <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
                    </>
                  )}
                </svg>
                Malignant probability
              </span>
              <span
                className="font-mono text-sm font-semibold tabular-nums"
                style={{ color: STATUS_COLOR[malignantStatus] }}
              >
                {(result.malignant_probability * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-[var(--status-critical-track)]">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: `${Math.max(result.malignant_probability * 100, 1)}%`,
                  background: STATUS_COLOR[malignantStatus],
                }}
              />
            </div>
            <p className="text-xs text-[var(--text-muted)]">
              Sum of MEL, BCC and AKIEC &mdash; the three classes treated as
              malignant/pre-malignant.
            </p>
          </div>

          {history.flagged && (
            <div
              className="flex flex-col gap-1.5 rounded-xl border-l-4 p-3.5"
              style={{
                borderColor: "var(--status-critical)",
                background: "color-mix(in srgb, var(--status-critical) 6%, var(--surface-2))",
              }}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-1.5 text-sm font-semibold text-[var(--status-critical)]">
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <circle cx="11" cy="11" r="7" />
                    <path d="m21 21-3.8-3.8" />
                  </svg>
                  Melanoma warning signs reported
                </span>
                <span className="shrink-0 rounded-full bg-[var(--surface-2)] px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
                  Glasgow 7-point rule &middot; not model output
                </span>
              </div>
              <p className="text-xs text-[var(--text-secondary)]">
                {(() => {
                  const hits = [...history.majorHits, ...history.minorHits].join("; ");
                  const context =
                    history.contextHits.length > 0
                      ? ` (${history.contextHits.join(", ")})`
                      : "";
                  return `${hits}${context}. Score ${history.score} — a history/symptom score of 3+ (major signs count double) is the standard threshold for suggesting specialist review. This model's melanoma sensitivity was the lowest of its three malignant classes on held-out test data (~71%), so these patient-reported signs are worth weighing even when the model's own top prediction isn't melanoma.`;
                })()}
              </p>
            </div>
          )}

          <div className="flex flex-col gap-2">
            {sortedProbs.map(([label, prob]) => {
              const info = CLASS_INFO[label];
              const color = info?.malignant ? "var(--status-critical)" : "var(--accent)";
              return (
                <div key={label} className="flex items-center gap-3 text-sm">
                  <span
                    className="h-2 w-2 shrink-0 rounded-full"
                    style={{ background: color }}
                  />
                  <span className="w-14 shrink-0 font-mono text-xs text-[var(--text-primary)]">
                    {label}
                  </span>
                  <span className="hidden w-40 shrink-0 truncate text-xs text-[var(--text-muted)] sm:block">
                    {info?.name}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-[var(--gridline)]">
                    <div
                      className="h-full rounded-full transition-all duration-700 ease-out"
                      style={{ width: `${Math.max(prob * 100, 1)}%`, background: color }}
                    />
                  </div>
                  <span className="w-14 shrink-0 text-right font-mono text-xs tabular-nums text-[var(--text-secondary)]">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>

          <div className="flex items-center gap-4 text-xs text-[var(--text-muted)]">
            <span className="flex items-center gap-1.5">
              <span className="h-2 w-2 rounded-full" style={{ background: "var(--accent)" }} />
              Benign
            </span>
            <span className="flex items-center gap-1.5">
              <span
                className="h-2 w-2 rounded-full"
                style={{ background: "var(--status-critical)" }}
              />
              Malignant (MEL &middot; BCC &middot; AKIEC)
            </span>
          </div>

          {result.scorecam_png_base64 && (
            <div className="flex flex-col gap-1.5 border-t border-[var(--border-hairline)] pt-4">
              <p className="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
                Score-CAM &mdash; predicted-class evidence
              </p>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={`data:image/png;base64,${result.scorecam_png_base64}`}
                alt="Score-CAM overlay"
                className="max-h-72 w-fit rounded-xl border border-[var(--border-hairline)] shadow-sm"
              />
            </div>
          )}
          </div>

          <p className="border-t border-[var(--border-hairline)] pt-3 text-xs text-[var(--text-muted)]">
            {result.disclaimer}
          </p>
        </div>
      )}
    </div>
  );
}
