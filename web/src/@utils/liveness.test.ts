import { describe, expect, it } from "vitest";
import type { LivenessProgressMessage, LivenessState } from "@interfaces/liveness";
import type { Guidance } from "@utils/camera";
import { describeFailure, livenessPrompt, progressFraction } from "@utils/liveness";

const message = (
  state: LivenessState,
  progress: [number, number] = [0, 11],
): LivenessProgressMessage => ({
  type: "progress",
  session_id: "s",
  state,
  progress,
  detection: null,
  eye: null,
});

const guidance = (message: string): Guidance => ({ message, ready: false });

describe("livenessPrompt", () => {
  it("asks the person to get into frame before any detection has arrived", () => {
    const prompt = livenessPrompt(message("AWAITING_FACE"));
    expect(prompt.message).toMatch(/frame/i);
    expect(prompt.active).toBe(false);
  });

  it("says WHY the hold has not started once a detection can tell it", () => {
    // A bar that will not move is useless on its own. The reason it will not
    // move is the only thing the person can act on.
    const prompt = livenessPrompt(
      message("AWAITING_FACE"),
      guidance("Move closer so your face is clear and detailed."),
    );
    expect(prompt.message).toBe("Move closer so your face is clear and detailed.");
    expect(prompt.active).toBe(false);
  });

  it("does not let stale guidance override the hold once it is running", () => {
    // The guidance and the session state arrive on the same message, but the
    // stabiliser damps the words — so guidance can lag a frame behind PASSED.
    const prompt = livenessPrompt(message("HOLDING"), guidance("No face in view."));
    expect(prompt.message).toMatch(/hold still/i);
    expect(prompt.active).toBe(true);
  });

  it("asks the person to hold still while the run accumulates", () => {
    const prompt = livenessPrompt(message("HOLDING", [4, 11]));
    expect(prompt.message).toMatch(/hold still/i);
    expect(prompt.active).toBe(true);
  });

  it("asks for the blink only once the service is ready to measure it", () => {
    // Asking earlier would waste the blink: the baseline it is compared against
    // is built from the frames that come before it.
    expect(livenessPrompt(message("HOLDING")).message).not.toMatch(/blink/i);
    expect(livenessPrompt(message("AWAITING_BLINK")).message).toMatch(/blink/i);
  });

  it("never instructs the person to move", () => {
    // The whole point of the passive rewrite. A prompt that asks for a turn
    // would be describing behaviour the backend no longer measures.
    const states: LivenessState[] = [
      "AWAITING_FACE",
      "HOLDING",
      "AWAITING_BLINK",
      "PASSED",
      "FAILED",
    ];
    for (const state of states) {
      expect(livenessPrompt(message(state)).message).not.toMatch(/turn|closer|left|right/i);
    }
  });

  it("never claims the subject was proved to be live", () => {
    // This check does not defeat a printed photo. Wording that implies it does
    // would misrepresent the guarantee to whoever reads the screen.
    const states: LivenessState[] = [
      "AWAITING_FACE",
      "HOLDING",
      "AWAITING_BLINK",
      "PASSED",
      "FAILED",
    ];
    for (const state of states) {
      expect(livenessPrompt(message(state)).message).not.toMatch(/liveness|live person|verified/i);
    }
  });

  it("reports the terminal states", () => {
    expect(livenessPrompt(message("PASSED")).message.length).toBeGreaterThan(0);
    expect(livenessPrompt(message("FAILED")).message).toMatch(/did not complete/i);
  });

  it("covers every state without producing undefined", () => {
    const states: LivenessState[] = [
      "AWAITING_FACE",
      "HOLDING",
      "AWAITING_BLINK",
      "PASSED",
      "FAILED",
    ];
    for (const state of states) {
      expect(livenessPrompt(message(state)).message).not.toMatch(/undefined/);
    }
  });

  it("has something to say before the first message arrives", () => {
    expect(livenessPrompt(null).message.length).toBeGreaterThan(0);
    expect(livenessPrompt(null).active).toBe(false);
  });
});

describe("describeFailure", () => {
  it("explains each failure in words the person can act on", () => {
    expect(describeFailure("FACE_LOST")).toMatch(/lost sight/i);
    expect(describeFailure("SESSION_EXPIRED")).toMatch(/timed out/i);
    expect(describeFailure("BLINK_NOT_SEEN")).toMatch(/blink/i);
  });

  it("never blames the person for an unknown failure", () => {
    const text = describeFailure(null);
    expect(text.length).toBeGreaterThan(0);
    expect(text).not.toMatch(/undefined/);
  });
});

describe("progressFraction", () => {
  it("maps completed steps onto 0..1", () => {
    expect(progressFraction([0, 11])).toBe(0);
    expect(progressFraction([11, 11])).toBe(1);
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
