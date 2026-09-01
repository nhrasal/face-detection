import { useEffect, useRef, useState } from "react";
import {
  INITIAL_GUIDANCE,
  type Guidance,
  frameGuidance,
  initialGuidanceState,
  stabiliseGuidance,
} from "@utils/camera";
import type { LiveDetection } from "@interfaces/face";

/**
 * The one instruction to show for a live detection, held steady.
 *
 * Detections arrive up to 30 times a second and a face sitting on a quality
 * threshold flips verdict between adjacent frames. Rendering that directly gives
 * strobing text and a button that flickers under the pointer, so a new verdict
 * has to hold for a few frames before it replaces the one on screen. The
 * overlay box still tracks every frame — only the words are damped.
 *
 * Shared by both camera panels so they cannot drift apart in what they say.
 */
export function useStableGuidance(detection: LiveDetection | null): Guidance {
  const [guidance, setGuidance] = useState<Guidance>(INITIAL_GUIDANCE);
  const stabiliser = useRef(initialGuidanceState());

  useEffect(() => {
    stabiliser.current = stabiliseGuidance(stabiliser.current, frameGuidance(detection));
    const { shown } = stabiliser.current;
    setGuidance((current) =>
      current.message === shown.message && current.ready === shown.ready ? current : shown,
    );
  }, [detection]);

  return guidance;
}
