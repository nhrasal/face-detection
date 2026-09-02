import { describe, expect, it } from "vitest";
import type { DetectResult, LiveDetection } from "@interfaces/face";
import {
  frameGuidance,
  initialGuidanceState,
  overlayBox,
  scaleToFit,
  stabiliseGuidance,
  visibleRegion,
} from "@utils/camera";

const frame = { width: 640, height: 480 };

const detection = (overrides: Partial<DetectResult> = {}): LiveDetection => ({
  frameWidth: frame.width,
  frameHeight: frame.height,
  result: {
    success: true,
    status: "OK",
    face_detected: true,
    face_count: 1,
    quality_score: 0.9,
    quality_issues: [],
    bounding_box: null,
    ...overrides,
  },
});

describe("scaleToFit", () => {
  it("shrinks the longest side to the limit and keeps the aspect ratio", () => {
    expect(scaleToFit(1920, 1080, 640)).toEqual({ width: 640, height: 360 });
  });
  it("never upscales a frame that is already small enough", () => {
    expect(scaleToFit(320, 240, 640)).toEqual({ width: 320, height: 240 });
  });
  it("survives a camera that has not reported a size yet", () => {
    expect(scaleToFit(0, 0, 640)).toEqual({ width: 0, height: 0 });
  });
});

describe("visibleRegion", () => {
  it("crops the sides of a landscape frame shown in a portrait box", () => {
    // 480 tall at 4:5 shows 384 of the 640 width, evenly trimmed at both ends.
    expect(visibleRegion(640, 480, 0.8)).toEqual({ x: 128, y: 0, width: 384, height: 480 });
  });

  it("crops the top and bottom of a frame taller than the box", () => {
    expect(visibleRegion(480, 640, 0.8)).toEqual({ x: 0, y: 20, width: 480, height: 600 });
  });

  it("shows the whole frame when no aspect is asked for", () => {
    expect(visibleRegion(640, 480)).toEqual({ x: 0, y: 0, width: 640, height: 480 });
  });

  it("returns nothing to show for a frame with no size", () => {
    expect(visibleRegion(0, 0, 0.8)).toEqual({ x: 0, y: 0, width: 0, height: 0 });
  });
});

describe("overlayBox", () => {
  const box = { x: 64, y: 48, width: 128, height: 96 };

  it("converts detector pixels to percentages of the preview", () => {
    expect(overlayBox(box, frame, false)).toEqual({ left: 10, top: 10, width: 20, height: 20 });
  });

  it("mirrors horizontally to match the mirrored preview", () => {
    // x=64 w=128 in a 640-wide frame leaves 448px to the right, so the mirrored
    // box starts at 70% — not at the 10% the unmirrored box uses.
    expect(overlayBox(box, frame, true)).toMatchObject({ left: 70, width: 20, top: 10 });
  });

  it("clips a box that runs past the frame edge instead of overflowing", () => {
    const overhang = overlayBox({ x: 600, y: 0, width: 200, height: 200 }, frame, false);
    expect(overhang?.left).toBeCloseTo(93.75);
    expect((overhang?.left ?? 0) + (overhang?.width ?? 0)).toBeLessThanOrEqual(100);
  });

  it("measures against the cropped view, not the frame the detector saw", () => {
    // The viewfinder shows the middle 384px of a 640px frame, so a box at x=256
    // sits a third of the way across what is actually on screen — not the fifth
    // of the way it would be across the full frame.
    const cropped = overlayBox({ x: 256, y: 48, width: 128, height: 96 }, frame, false, 0.8);
    expect(cropped?.left).toBeCloseTo(33.33, 1);
    expect(cropped?.width).toBeCloseTo(33.33, 1);
    expect(cropped?.top).toBeCloseTo(10);
  });

  it("clips a box that falls outside the cropped view", () => {
    const offscreen = overlayBox({ x: 0, y: 48, width: 64, height: 96 }, frame, false, 0.8);
    expect(offscreen?.left).toBe(0);
    expect(offscreen?.width).toBe(0);
  });

  it("returns nothing when the frame has no size", () => {
    expect(overlayBox(box, { width: 0, height: 0 }, false)).toBeNull();
  });
});

describe("frameGuidance", () => {
  it("waits before claiming anything about an unmeasured frame", () => {
    expect(frameGuidance(null)).toEqual({ message: "Looking for a face…", ready: false });
  });

  it("names no control, because only one of the two panels has one", () => {
    // The camera panel has a shutter; the liveness panel captures itself on the
    // blink. Both render this message, so it cannot mention pressing anything
    // without sending liveness subjects looking for a button that is not there.
    expect(frameGuidance(detection()).message).not.toMatch(/capture|press|button|tap|click/i);
  });

  it("is ready only when the service says the frame is OK", () => {
    expect(frameGuidance(detection()).ready).toBe(true);
  });

  it("asks for a face when none is in view", () => {
    const guidance = frameGuidance(detection({ status: "NO_FACE", face_count: 0, face_detected: false }));
    expect(guidance).toEqual({ message: "No face in view. Center your face in the frame.", ready: false });
  });

  it("blocks capture on a second face, which verification would reject anyway", () => {
    const guidance = frameGuidance(detection({ face_count: 2 }));
    expect(guidance.ready).toBe(false);
    expect(guidance.message).toContain("More than one person");
  });

  it("speaks the same words as the verification result for a quality issue", () => {
    const guidance = frameGuidance(
      detection({ status: "LOW_QUALITY", quality_issues: ["FACE_TOO_SMALL", "IMAGE_BLURRY"] }),
    );
    // One instruction at a time: the first issue only, not all of them.
    expect(guidance.message).toBe("Move closer so your face is clear and detailed.");
    expect(guidance.ready).toBe(false);
  });

  it("still gives an instruction for a reason code it does not recognise", () => {
    const guidance = frameGuidance(detection({ status: "LOW_QUALITY", quality_issues: ["SOMETHING_NEW"] }));
    expect(guidance.message).toBe("Adjust your position and lighting.");
    expect(guidance.ready).toBe(false);
  });
});

describe("stabiliseGuidance", () => {
  const ready = { message: "Hold still — ready to capture.", ready: true };
  const closer = { message: "Move closer so your face is clear and detailed.", ready: false };

  const feed = (values: Array<typeof ready>, streak = 3) =>
    values.reduce((state, next) => stabiliseGuidance(state, next, streak), initialGuidanceState());

  it("ignores a verdict that has not held long enough", () => {
    // Two frames of "ready" among a run of "move closer" must not arm the button.
    expect(feed([closer, closer, closer, ready, ready]).shown).toEqual(closer);
  });

  it("adopts a verdict once it holds for the full streak", () => {
    expect(feed([closer, closer, closer, ready, ready, ready]).shown).toEqual(ready);
  });

  it("resets the run when the verdict flaps back and forth", () => {
    // Alternating frames never accumulate a run, so nothing ever changes.
    expect(feed([ready, closer, ready, closer, ready, closer]).shown).toEqual({
      message: "Looking for a face…",
      ready: false,
    });
  });

  it("holds the current verdict steady while it keeps being confirmed", () => {
    const settled = feed([ready, ready, ready]);
    expect(stabiliseGuidance(settled, ready).shown).toEqual(ready);
    expect(stabiliseGuidance(settled, ready).streak).toBe(0);
  });
});
