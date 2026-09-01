import { describe, expect, it } from "vitest";
import type { ChallengeKind, LivenessProgressMessage, LivenessState } from "@interfaces/liveness";
import { describeFailure, livenessPrompt, progressFraction } from "@utils/liveness";

const message = (
  state: LivenessState,
  challenge: ChallengeKind | null = null,
  progress: [number, number] = [0, 3],
): LivenessProgressMessage => ({
  type: "progress",
  session_id: "s",
  state,
  challenge,
  progress,
});

describe("livenessPrompt", () => {
  it("asks the person to look straight while awaiting a neutral frame", () => {
    // The backend requires a frontal frame between challenges. Without telling
    // the person, a passing attempt looks like a stuck screen.
    const prompt = livenessPrompt(message("AWAITING_NEUTRAL"));
    expect(prompt.message).toMatch(/straight/i);
    expect(prompt.settle).toBe(true);
  });

  it("gives the movement instruction while a challenge is active", () => {
    const prompt = livenessPrompt(message("AWAITING_CHALLENGE", "TURN_LEFT"));
    expect(prompt.message).toMatch(/turn your head to your left/i);
    expect(prompt.settle).toBe(false);
  });

  it("phrases turns about the person's own body, not the image", () => {
    // The preview is mirrored, so an image-relative instruction would be
    // backwards for the user while looking correct in code.
    expect(livenessPrompt(message("AWAITING_CHALLENGE", "TURN_LEFT")).message).toMatch(/your left/i);
    expect(livenessPrompt(message("AWAITING_CHALLENGE", "TURN_RIGHT")).message).toMatch(
      /your right/i,
    );
  });

  it("covers every challenge kind", () => {
    const kinds: ChallengeKind[] = ["TURN_LEFT", "TURN_RIGHT", "MOVE_CLOSER"];
    for (const kind of kinds) {
      const prompt = livenessPrompt(message("AWAITING_CHALLENGE", kind));
      expect(prompt.message.length).toBeGreaterThan(0);
      expect(prompt.message).not.toMatch(/undefined/);
    }
  });

  it("falls back when a challenge is active but unnamed", () => {
    expect(livenessPrompt(message("AWAITING_CHALLENGE", null)).message).toMatch(/instruction/i);
  });

  it("reports the terminal states", () => {
    expect(livenessPrompt(message("PASSED")).message).toMatch(/confirmed/i);
    expect(livenessPrompt(message("FAILED")).message).toMatch(/failed/i);
  });

  it("has something to say before the first message arrives", () => {
    expect(livenessPrompt(null).message.length).toBeGreaterThan(0);
  });
});

describe("describeFailure", () => {
  it("explains each failure in words the person can act on", () => {
    expect(describeFailure("CHALLENGE_TIMEOUT")).toMatch(/too long/i);
    expect(describeFailure("FACE_LOST")).toMatch(/lost sight/i);
    expect(describeFailure("SESSION_EXPIRED")).toMatch(/timed out/i);
  });

  it("never blames the person for an unknown failure", () => {
    const text = describeFailure(null);
    expect(text.length).toBeGreaterThan(0);
    expect(text).not.toMatch(/undefined/);
  });
});

describe("progressFraction", () => {
  it("maps completed challenges onto 0..1", () => {
    expect(progressFraction([0, 3])).toBe(0);
    expect(progressFraction([3, 3])).toBe(1);
    expect(progressFraction([1, 4])).toBeCloseTo(0.25);
  });

  it("never divides by zero", () => {
    expect(progressFraction([0, 0])).toBe(0);
  });

  it("clamps nonsense into range", () => {
    expect(progressFraction([9, 3])).toBe(1);
    expect(progressFraction([-1, 3])).toBe(0);
  });
});
