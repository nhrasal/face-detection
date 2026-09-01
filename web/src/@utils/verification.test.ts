import { describe, expect, it } from "vitest";
import type { VerificationResult } from "@interfaces/face";
import { asPercentage, resultPresentation, similarityLabel, validatePhoto } from "@utils/verification";

const result = (overrides: Partial<VerificationResult> = {}): VerificationResult => ({
  matched: false, similarity: null, confidence: null, threshold: 0.363,
  decision: "LOW_QUALITY", reason_code: "CANDIDATE_LOW_QUALITY", quality_issues: [],
  reference_status: "OK", candidate_status: "LOW_QUALITY", processing_time_ms: 24, ...overrides,
});

describe("photo validation", () => {
  it("accepts supported photos", () => expect(validatePhoto(new File(["photo"], "face.jpg", { type: "image/jpeg" }))).toBeNull());
  it("rejects unsupported types", () => expect(validatePhoto(new File(["x"], "face.gif", { type: "image/gif" }))).toContain("JPEG"));
  it("rejects empty and oversized files", () => {
    expect(validatePhoto(new File([], "empty.png", { type: "image/png" }))).toContain("empty");
    expect(validatePhoto(new File([new Uint8Array(5 * 1024 * 1024 + 1)], "huge.png", { type: "image/png" }))).toContain("5 MB");
  });
});

describe("result messaging", () => {
  it("distinguishes confidence from raw similarity", () => { expect(asPercentage(0.823)).toBe("82%"); expect(similarityLabel(0.411234)).toBe("0.4112"); });
  it("provides actionable quality guidance", () => {
    const copy = resultPresentation(result({ quality_issues: ["IMAGE_BLURRY", "TOO_DARK", "EXTREME_POSE"] }));
    expect(copy.guidance).toHaveLength(3); expect(copy.guidance.join(" ")).toMatch(/focus/i); expect(copy.guidance.join(" ")).toMatch(/light/i); expect(copy.guidance.join(" ")).toMatch(/straight/i);
  });
  it("renders dedicated decision outcomes", () => {
    expect(resultPresentation(result({ decision: "NO_FACE" })).title).toMatch(/No face/);
    expect(resultPresentation(result({ decision: "MULTIPLE_FACES" })).title).toMatch(/one face/);
    expect(resultPresentation(result({ decision: "REVIEW" })).title).toMatch(/review/i);
    expect(resultPresentation(result({ decision: "NO_MATCH" })).tone).toBe("danger");
  });
});
