import type { BoundingBox, LiveDetection } from "@interfaces/face";
import { issueMessages } from "@utils/verification";

/** Live guidance frames are small on purpose: they are thrown away after one detection. */
export const FRAME_MAX_SIDE = 640;
export const FRAME_QUALITY = 0.7;

/** The still that is actually verified keeps enough detail for the recogniser. */
export const CAPTURE_MAX_SIDE = 1280;
export const CAPTURE_QUALITY = 0.92;

/** Roughly 2.5 FPS — the HTTP fallback stays inside the roadmap's 2-5 FPS band. */
export const FRAME_INTERVAL_MS = 400;

/**
 * How long a frame must stay good before it is captured automatically.
 *
 * `stabiliseGuidance` already suppresses single-frame flicker, but three frames
 * is under a tenth of a second at full rate — quick enough to snap someone who
 * is still settling into position. This dwell is for the person, not the
 * detector: long enough to feel deliberate, short enough not to feel stuck.
 */
export const AUTO_CAPTURE_HOLD_MS = 800;

/**
 * Ceiling for the live stream: 30 FPS.
 *
 * The socket is ack-paced, so it would otherwise run as fast as the round trip
 * allows. Past 30 FPS nothing looks smoother to a person, while every extra
 * frame costs another inference on a shared worker — so the cap buys nothing
 * visible and doubles the server cost without it.
 */
export const MIN_STREAM_INTERVAL_MS = 1000 / 30;

export interface FrameSize {
  width: number;
  height: number;
}

/** Fit a frame inside `maxSide` without ever upscaling it. */
export function scaleToFit(width: number, height: number, maxSide: number): FrameSize {
  const longest = Math.max(width, height);
  if (longest <= maxSide || longest === 0) return { width, height };
  const scale = maxSide / longest;
  return { width: Math.max(1, Math.round(width * scale)), height: Math.max(1, Math.round(height * scale)) };
}

/**
 * Encode the video's current frame as a JPEG.
 *
 * Returns null while the camera is still negotiating a resolution — `videoWidth`
 * is 0 until then, and a zero-sized canvas encodes to a blob the API rejects.
 */
export async function grabFrame(
  video: HTMLVideoElement,
  maxSide: number,
  quality: number,
): Promise<{ blob: Blob; width: number; height: number } | null> {
  if (!video.videoWidth || !video.videoHeight) return null;
  const { width, height } = scaleToFit(video.videoWidth, video.videoHeight, maxSide);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  // The raw video, never the mirrored preview: mirroring is a comfort affordance
  // for the operator, not something that belongs in a stored or verified image.
  context.drawImage(video, 0, 0, width, height);
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", quality),
  );
  return blob ? { blob, width, height } : null;
}

export const frameToFile = (blob: Blob, name = "camera-capture.jpg"): File =>
  new File([blob], name, { type: "image/jpeg" });

export interface OverlayBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Place a detection box over the preview, as percentages of the preview element.
 *
 * The box arrives in the coordinates of the frame that was uploaded, so the
 * preview must be rendered at that same aspect ratio for this to line up. When
 * the preview is mirrored the box has to be mirrored with it, or it tracks the
 * face in the wrong direction.
 */
export function overlayBox(
  box: BoundingBox,
  frame: FrameSize,
  mirrored: boolean,
): OverlayBox | null {
  if (frame.width <= 0 || frame.height <= 0) return null;
  const clamp = (value: number) => Math.min(100, Math.max(0, value));
  const left = mirrored ? frame.width - (box.x + box.width) : box.x;
  const leftPercent = clamp((left / frame.width) * 100);
  const topPercent = clamp((box.y / frame.height) * 100);
  return {
    left: leftPercent,
    top: topPercent,
    // Detector boxes can run past the frame edge; clip rather than overflow.
    width: clamp((box.width / frame.width) * 100 + leftPercent) - leftPercent,
    height: clamp((box.height / frame.height) * 100 + topPercent) - topPercent,
  };
}

export interface Guidance {
  message: string;
  /** Whether this frame is worth capturing — the same bar verification applies. */
  ready: boolean;
}

/**
 * Turn one live detection into a single instruction.
 *
 * Only the first quality issue is shown. A list of four things to fix at 2.5 FPS
 * is noise; one instruction at a time is something a person can act on.
 */
export function frameGuidance(detection: LiveDetection | null): Guidance {
  if (!detection) return { message: "Looking for a face…", ready: false };
  const { result } = detection;

  if (result.face_count === 0) {
    return { message: "No face in view. Center your face in the frame.", ready: false };
  }
  // The live endpoint tracks the largest face so the overlay does not flicker when
  // someone walks past, but a second face still fails verification — say so now
  // rather than after the capture is rejected.
  if (result.face_count > 1) {
    return { message: "More than one person is in view.", ready: false };
  }
  if (result.status !== "OK") {
    const issue = result.quality_issues[0];
    return {
      message: (issue && issueMessages[issue]) || "Adjust your position and lighting.",
      ready: false,
    };
  }
  // No instruction to capture: both camera panels take the photo themselves
  // once this has held. Saying "ready to capture" would advertise a button that
  // is not there.
  return { message: "Hold still.", ready: true };
}

export const INITIAL_GUIDANCE: Guidance = { message: "Looking for a face…", ready: false };

export interface GuidanceState {
  shown: Guidance;
  candidate: Guidance | null;
  streak: number;
}

export const initialGuidanceState = (): GuidanceState => ({
  shown: INITIAL_GUIDANCE,
  candidate: null,
  streak: 0,
});

/**
 * Hysteresis for guidance arriving 30 times a second.
 *
 * Per-frame truth is not per-frame *advice*: a face hovering on the quality
 * threshold flips verdict between adjacent frames, and rendering that directly
 * gives a message that strobes — and, now that capture is automatic, a shutter
 * that fires on a frame the next one would have rejected. A new verdict has to
 * hold for `streak` consecutive frames before it replaces the one on screen, so
 * the box stays real-time while the words stay readable.
 */
export function stabiliseGuidance(
  state: GuidanceState,
  next: Guidance,
  streak = 3,
): GuidanceState {
  const same = (a: Guidance, b: Guidance) => a.message === b.message && a.ready === b.ready;

  if (same(next, state.shown)) return { shown: state.shown, candidate: null, streak: 0 };

  const runLength = state.candidate && same(state.candidate, next) ? state.streak + 1 : 1;
  if (runLength >= streak) return { shown: next, candidate: null, streak: 0 };
  return { shown: state.shown, candidate: next, streak: runLength };
}
