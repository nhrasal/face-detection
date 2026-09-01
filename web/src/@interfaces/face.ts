export type Decision = "MATCH" | "NO_MATCH" | "REVIEW" | "NO_FACE" | "MULTIPLE_FACES" | "LOW_QUALITY" | "PROCESSING_ERROR";
export interface User { id: string; external_id: string; name: string; profile_image_url: string; }
export interface VerificationResult {
  matched: boolean; similarity: number | null; confidence: number | null; threshold: number;
  decision: Decision; reason_code: string | null; quality_issues: string[];
  reference_status: string; candidate_status: string; processing_time_ms: number;
}
export interface ApiErrorBody {
  error?: { code?: string; message?: string; detail?: string | null };
  detail?: string | Array<{ msg?: string }>;
}
